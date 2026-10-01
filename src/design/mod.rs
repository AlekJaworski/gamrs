//! Design strategies — typed fit-time basis preparation + predict-time
//! design rebuilder.
//!
//! Two surfaces:
//! - [`DesignStrategy`] — open trait. Users could implement custom
//!   strategies, but the canonical set ships in this module (`Cr`, `Re`,
//!   `CrStable`, `Additive`). Consumed at fit-time, not stored.
//! - [`Predictor`] — closed `enum`. Carries whatever the strategy needs
//!   to rebuild a design matrix on new x at predict-time. Library-
//!   controlled (users extend Loss/Link/VarianceFn, not bases) so the
//!   match-dispatch surface is closed — **no `Box<dyn>`** per
//!   `feedback_enum_vs_trait_dispatch`.
//!
//! Compose-via-types: `Cr { k }` builds a `Predictor::Cr(CrPredictor)`,
//! `CrStable { k }` builds a `Predictor::CrStable(CrStablePredictor)` that
//! carries the stable-reparam rotation V on top of the CR centring,
//! `Additive { terms }` builds a `Predictor::Additive(AdditivePredictor)`
//! that delegates per-term reconstruction to a `Vec<Predictor>`. The
//! [`FittedGam`](crate::fit::FittedGam) `predict` / `predict_deriv`
//! methods delegate via `match self.predictor`.
//!
//! Multi-smooth (94b) layout: `Additive { terms: Vec<TermSpec> }` builds
//! a design `[1 | X_1 | X_2 | …]` with each term's centred basis stacked
//! horizontally, plus a `Vec<S_j>` of `(p, p)` block-embedded penalties
//! (one per smoothing parameter). The single-smooth path through
//! `Cr / Re / CrStable` produces `s_list.len() == 1`.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::Eigh;

use crate::error::{GamrsError, Result};

mod additive;
mod cr;
pub(crate) mod identifiability;
mod parametric;
mod re;
mod tensor;
mod tps;

pub use additive::{Additive, AdditivePredictor, MarginKind, TermSpec};
pub use cr::{Cr, CrPredictor, CrStable, CrStablePredictor};
pub use parametric::ParametricPredictor;
pub use re::{Re, RePredictor};
pub use tensor::{TensorMultiPredictor, TensorPredictor};
pub use tps::TpsPredictor;

// =============================================================================
// Predictor — predict-time design rebuilder. Closed-set `enum`, NOT
// `Box<dyn>`: zero-cost match dispatch. New basis types extend the enum
// (a library-controlled change), not user code.
// =============================================================================

/// Predict-time design rebuilder. Library-controlled closed set —
/// `match` dispatch is zero-cost. See module docs for rationale.
///
/// Serde representation (under the `persistence` feature) uses the
/// default externally-tagged form — wire-format compatible with both
/// bincode (binary) and serde_json (text). Bincode does not support
/// `serde(tag = "...")` because it requires `deserialize_any`, so the
/// internally-tagged form would only work for the JSON path.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum Predictor {
    /// CR spline + sum-to-zero centring + intercept column.
    Cr(CrPredictor),
    /// Random-effect one-hot lookup + intercept column.
    Re(RePredictor),
    /// CR spline + sum-to-zero centring + `StableReparam` rotation V +
    /// intercept column. Predictions are basis-invariant w.r.t. the
    /// unrotated `Cr` path up to FP.
    CrStable(CrStablePredictor),
    /// Additive multi-smooth: `[1 | X_1 | X_2 | …]` with per-term
    /// predictors delegating to `Cr` / `Re` / `CrStable` / `Tensor`.
    Additive(AdditivePredictor),
    /// Tensor product `te(x_{col_a}, x_{col_b})` — anisotropic 2-margin
    /// product of centred CR margins, with the intercept column
    /// prepended. Two penalties per term.
    Tensor(TensorPredictor),
    /// N-margin tensor product `te(...)` or interaction `ti(...)` —
    /// anisotropic product of CR margins (uncentred for `te`, per-margin
    /// sum-to-zero for `ti`), intercept column prepended. D penalties.
    TensorMulti(TensorMultiPredictor),
    /// 2-D (or higher) isotropic thin-plate regression spline.
    /// Single penalty per term.
    Tps(TpsPredictor),
    /// Parametric (unsmoothed linear) term — single raw column, no
    /// penalty. Only used as a sub-Predictor of `Additive`.
    Parametric(ParametricPredictor),
}

impl Predictor {
    /// Rebuild the design matrix on new x. `x_new` has shape
    /// `(n_new, n_input_dims)`; each variant reads the columns it owns.
    /// Returns shape `(n_new, p)` where `p = 1 + Σ_j smooth_dim_j`
    /// (intercept column always present).
    pub fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        match self {
            Self::Cr(p) => p.design(x_new),
            Self::Re(p) => p.design(x_new),
            Self::CrStable(p) => p.design(x_new),
            Self::Additive(p) => p.design(x_new),
            Self::Tensor(p) => p.design(x_new),
            Self::TensorMulti(p) => p.design(x_new),
            Self::Tps(p) => p.design(x_new),
            Self::Parametric(p) => p.design(x_new),
        }
    }

    /// Rebuild `∂design/∂x_axis` on new x. The intercept column always
    /// contributes a zero column. Caller composes with `β_smooth` to get
    /// `∂μ̂/∂x`.
    pub fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        match self {
            Self::Cr(p) => p.design_deriv(x_new, axis),
            Self::Re(p) => p.design_deriv(x_new, axis),
            Self::CrStable(p) => p.design_deriv(x_new, axis),
            Self::Additive(p) => p.design_deriv(x_new, axis),
            Self::Tensor(p) => p.design_deriv(x_new, axis),
            Self::TensorMulti(p) => p.design_deriv(x_new, axis),
            Self::Tps(p) => p.design_deriv(x_new, axis),
            Self::Parametric(p) => p.design_deriv(x_new, axis),
        }
    }
}

// =============================================================================
// PreparedDesign — what every DesignStrategy returns at fit-time.
// =============================================================================

/// Fit-time output of a [`DesignStrategy`]. Carries the design matrix,
/// the per-term penalty list, rank diagnostics, and the [`Predictor`]
/// that will rebuild new designs at predict-time.
///
/// Multi-smooth (94b/94c): `s_list` is `Vec<S_j>` of `(p, p)` block-
/// embedded penalty matrices (one per smoothing parameter ρ_j).
/// Single-smooth strategies (`Cr`, `Re`, `CrStable`) produce
/// `s_list.len() == 1`; `Additive { terms }` produces a flat list with
/// 1 penalty per univariate term and 2 penalties per tensor term —
/// `s_list.len() = Σ_j n_smoothing_params(term_j)`.
/// The total penalty seen by the inner solver at smoothing parameter
/// vector ρ is `S_total(ρ) = Σ_j exp(ρ_j) · s_list[j]`.
pub struct PreparedDesign {
    /// `(n, p)` design with intercept column at index 0.
    pub x_design: Array2<f64>,
    /// `Vec<(p, p)>` — one block-embedded penalty per smoothing parameter.
    /// Each matrix has the term's penalty in its column block and zeros
    /// elsewhere (so addition assembles a block-diagonal `S_total`).
    /// Intercept row/column is zero in every block.
    pub s_list: Vec<Array2<f64>>,
    /// Per-term rank of `s_list[j]` (count of strictly positive
    /// eigenvalues, above a relative tolerance).
    pub rank_s_list: Vec<usize>,
    /// Per-term `log|S_j|+` at `λ_j = 1` (sum of log positive
    /// eigenvalues). Score uses `Σ_j (rank_j · ρ_j + log_pseudo_det_j)`
    /// for the `log|λS|+` term.
    pub log_pseudo_det_s_list: Vec<f64>,
    /// Null-space dimension of the total penalty:
    /// `Mp = p − Σ_j rank_s_list[j]`.
    pub mp: usize,
    /// Predict-time rebuilder. Closed-set `enum`; **no `Box<dyn>`**.
    pub predictor: Predictor,
}

// =============================================================================
// DesignStrategy — open trait. `Cr` / `Re` / `CrStable` / `Additive` are
// the canonical impls.
// =============================================================================

/// Fit-time design preparation. Open trait (users *could* implement
/// custom strategies), but the canonical [`Predictor`] enum it produces
/// is closed. See module docs.
///
/// `x` has shape `(n_obs, n_input_dims)`. Single-smooth impls (`Cr`,
/// `Re`, `CrStable`) read `x.column(0)`; `Additive` reads each term's
/// configured column index.
pub trait DesignStrategy {
    fn prepare(&self, x: ArrayView2<f64>) -> Result<PreparedDesign>;
}

// =============================================================================
// Small math helpers — local to design/; previously in design.rs.
// =============================================================================

/// Prepend a constant `1.0` intercept column to a `(n, k)` design.
pub(crate) fn prepend_intercept(centred: ArrayView2<f64>) -> Array2<f64> {
    let n = centred.nrows();
    let k = centred.ncols();
    let mut x_design = Array2::<f64>::zeros((n, 1 + k));
    for i in 0..n {
        x_design[[i, 0]] = 1.0;
        for j in 0..k {
            x_design[[i, 1 + j]] = centred[[i, j]];
        }
    }
    x_design
}

/// Prepend a zero column to a derivative `(n, k)` matrix — the
/// intercept's contribution to ∂design/∂x is identically zero.
pub(crate) fn prepend_zero_column(centred_d1: ArrayView2<f64>) -> Array2<f64> {
    let n = centred_d1.nrows();
    let k = centred_d1.ncols();
    let mut d = Array2::<f64>::zeros((n, 1 + k));
    for i in 0..n {
        for j in 0..k {
            d[[i, 1 + j]] = centred_d1[[i, j]];
        }
    }
    d
}

pub(crate) fn matrix_inf_norm(m: ArrayView2<f64>) -> f64 {
    let mut max_row = 0.0_f64;
    for r in m.axis_iter(Axis(0)) {
        let s: f64 = r.iter().map(|v| v.abs()).sum();
        if s > max_row {
            max_row = s;
        }
    }
    max_row
}

/// Rank + `log|S|+` of a symmetric PSD penalty matrix. The pseudo-det is
/// the product of strictly positive eigenvalues; the rank counts them
/// against a relative tolerance.
///
/// Returns the mathematically-correct count of positive eigenvalues.
/// Some downstream score paths (ocat in particular) apply a per-family
/// "mgcv heuristic" rank adjustment via `Loss::score_rank_adjustment`
/// to match mgcv's `non_zero_rows − 2` convention for CR splines.
pub(crate) fn rank_and_log_pseudo_det(s: ArrayView2<f64>) -> Result<(usize, f64)> {
    let s_owned = s.to_owned();
    // eigh on a symmetric penalty is robust but can fail on pathological
    // user-supplied designs (LAPACK convergence). Surface it as a typed
    // GamrsError instead of a panic so the Python wheel raises a catchable
    // exception with a clear message rather than a `PanicException`.
    let (eigs, _) = s_owned
        .eigh(ndarray_linalg::UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(format!("penalty eigendecomposition failed: {e}")))?;
    let max_eig = eigs.iter().cloned().fold(0.0_f64, f64::max);
    let tol = max_eig.max(1.0) * 1e-10;
    let mut rank = 0usize;
    let mut log_det = 0.0_f64;
    for &e in eigs.iter() {
        if e > tol {
            rank += 1;
            log_det += e.ln();
        }
    }
    Ok((rank, log_det))
}

/// Assemble `S_total(ρ) = Σ_j exp(ρ_j) · s_list[j]`. The hot path inside
/// every inner solver — single allocation, sequential add.
///
/// `p` is the design width, and is a parameter rather than `s_list[0].nrows()` because an
/// all-parametric design has NO penalties: the width cannot be read off a penalty that does not
/// exist, and an empty `s_list` must yield a zero penalty of the right shape rather than panic.
/// Callers all hold the design, so they can answer this; the alternative — a
/// `s_list.is_empty()` guard at each of the eleven call sites — is the same fix paid for eleven
/// times. The lasting shape is a `Penalties` type owning both the list and the width; see the
/// design note.
pub fn combined_s(s_list: &[Array2<f64>], rho: &Array1<f64>, p: usize) -> Array2<f64> {
    debug_assert_eq!(
        s_list.len(),
        rho.len(),
        "combined_s: s_list and rho length mismatch"
    );
    let mut s_total = Array2::<f64>::zeros((p, p));
    for (j, s_j) in s_list.iter().enumerate() {
        let lambda = rho[j].exp();
        for r in 0..p {
            for c in 0..p {
                s_total[[r, c]] += lambda * s_j[[r, c]];
            }
        }
    }
    s_total
}
