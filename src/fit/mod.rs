//! Layer 6 — end-to-end orchestrator: data → DesignStrategy → Inner →
//! Score → Outer → FittedGam → Predictor.
//!
//! The single public Rust entry point is `gamrs::fit(family, x, y, w, k)`
//! plus its `fit_with_design` / `fit_with_solver` / `fit_with` typed
//! extensions. Dispatch is type-driven via the `FamilyFitWithSolver`
//! trait — no string keys, no runtime branching, no per-family
//! `fit_*_cr` wrappers.
//!
//! Every `FamilyFitWithSolver::fit_with_solver_canonical` body follows
//! the same skeleton:
//!
//! 1. Validate `x` / `y` / `weights` lengths and family-specific support.
//! 2. Build the design + total penalty via `DesignStrategy::prepare`
//!    (defaults to `Cr { k }`).
//! 3. Build the inner solver + envelope score for the family.
//! 4. Run the outer Newton on θ.
//! 5. Re-run the inner at θ̂ for the reported `FittedGam`.
//! 6. EDF = `tr(H⁻¹ · X'WX)`; report `scale` per family convention.
//!
//! Steps 1, 2, 5(b), 6 are mechanical; the family-specific work is the
//! Family/Inner/Score wiring + the y-validation + the `scale` formula.
//! `fit_pirls_envelope` / `fit_shape_aware` (in `driver.rs`) capture the
//! mechanical shell so each impl reduces to "validate + wire + scale".
//!
//! Split layout (by role):
//! - `mod.rs`       — `FittedGam`, validation helpers.
//! - `driver.rs`    — `fit_pirls_envelope` + `fit_shape_aware` generic
//!                    drivers shared by every family.
//! - `gaussian.rs`  — `fit_gaussian_from_prep` helper (closed-form REML).
//! - `quantile.rs`  — `fit_quantile_from_prep` helper (qgam warm start +
//!                    ArmijoElfInner).
//! - `canonical.rs` — `FamilyFitWithSolver`/`FamilyFit`/`FitWithProfile`
//!                    dispatch traits + per-family impls + `fit` /
//!                    `fit_with` / `fit_with_solver` / `fit_with_design`
//!                    public functions.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::design::Predictor;
use crate::error::{GamrsError, Result};

pub mod canonical;
pub mod driver;
mod family_impls;
pub mod gaulss;
// scat response-standardization scale, shared with the Python wrapper's
// scat/tdist dispatch arms so the default σ² seed matches the in-core fit.
// Only the Python bindings consume the re-export; family_impls uses the fn
// directly, so gate the re-export to avoid an unused-import warning otherwise.
#[cfg(feature = "python")]
pub(crate) use family_impls::scat_response_scale;
pub mod gaussian;
#[cfg(feature = "persistence")]
mod persistence;
pub(crate) mod profile_shape;
pub mod quantile;
pub mod shash;

pub use canonical::{
    fit, fit_with, fit_with_design, fit_with_solver, FamilyFit, FamilyFitWithSolver, FitWithProfile,
};
pub use driver::{ExplicitInit, LambdaInit, MgcvInit, SmartInit, ZeroInit};
pub use gaulss::{fit_gaulss, GaulssFit, GaulssOpts};
pub use shash::{fit_shash, ShashGamFit, ShashGamOpts};

/// Link kind tag carried on the [`FittedGam`] so `predict_ci` /
/// `predict_diff` can map between η (linear-predictor) and μ (response)
/// scales without re-importing the family type at the call site.
///
/// Closed-set enum (no `Box<dyn>`) — every family in gamrs uses one of
/// these canonical link kinds. New families adopting a new link kind
/// extend this enum (a library-controlled change).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum LinkKind {
    Identity,
    Log,
    Logit,
    /// Reciprocal link `η = 1/μ` (mgcv's canonical link for Gamma).
    Inverse,
}

impl LinkKind {
    /// `g⁻¹(η)` — apply the inverse link to a single value.
    pub fn inverse(self, eta: f64) -> f64 {
        match self {
            Self::Identity => eta,
            Self::Log => eta.exp(),
            Self::Logit => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
            Self::Inverse => {
                // μ = 1/η. Floor |η| away from 0.
                let eps = 1e-300;
                let sign = if eta < 0.0 { -1.0 } else { 1.0 };
                1.0 / (sign * eta.abs().max(eps))
            }
        }
    }

    /// `dμ/dη` — derivative of the inverse link wrt η (used by the
    /// delta-method on response-scale CIs).
    pub fn d_inverse(self, eta: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Log => eta.exp(),
            Self::Logit => {
                let mu = self.inverse(eta);
                mu * (1.0 - mu)
            }
            Self::Inverse => {
                // dμ/dη = -1/η²
                let eps = 1e-300;
                let e2 = (eta * eta).max(eps);
                -1.0 / e2
            }
        }
    }
}

/// Output scale for `predict_ci`. Closed-set enum per project standard
/// (no `Box<dyn>` for closed abstractions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictScale {
    /// Linear-predictor (η) scale — no inverse-link transform.
    Link,
    /// Response (μ) scale — apply `g⁻¹` to the CI endpoints, with the
    /// SE propagated through the delta method.
    Response,
}

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedGam {
    /// Coefficient vector: `[intercept, term_1 coefs, term_2 coefs, …]`.
    /// Layout depends on the [`Predictor`] (CR centred basis, RE one-hot,
    /// CR-stable rotated basis, or additive multi-smooth).
    pub beta: Array1<f64>,
    /// Fitted log smoothing parameters, one per smoothing term. Length 1
    /// for single-smooth fits (`Cr` / `Re` / `CrStable`), length T for
    /// `Additive { terms }` fits.
    pub rho: Array1<f64>,
    /// `λ_j = exp(ρ_j)` per term — convenience cache so downstream
    /// callers don't have to `exp` the rho each time.
    pub lambda: Array1<f64>,
    /// Profiled scale (σ̂²) for the Gaussian model — family-dependent.
    pub scale: f64,
    /// Effective dof = `tr(H⁻¹ X'WX)`.
    pub edf_total: f64,
    /// Per-term effective dof: `edf_per_term[j] = tr(H⁻¹ X'WX_j)` =
    /// `dim(term_j) - λ_j · tr(H⁻¹ S_j)`. Sums to `edf_total - 1` (the
    /// intercept contributes a fixed dof of 1). Length matches `rho`.
    pub edf_per_term: Array1<f64>,
    pub n: usize,
    pub n_iters: usize,
    pub converged: bool,
    /// REML value at the optimum (for diagnostics).
    pub reml_value: f64,
    /// Predict-time design rebuilder. Library-controlled closed-set
    /// `enum` — zero-cost match dispatch. See [`Predictor`].
    pub predictor: Predictor,
    /// Posterior covariance of β̂ on the fit basis: `σ̂² · A⁻¹`, where
    /// `A = X' diag(W) X + λS` is the converged penalised Hessian. For
    /// fixed-φ families (Bernoulli, Poisson, NegBin, scat, ocat, ELF)
    /// `σ̂² = 1`; for profiled-φ families this is the Pearson / mgcv-two-
    /// sigma estimate. Computed once at fit-time via
    /// [`crate::inner::LinearSolver::invert`].
    pub vcov: Array2<f64>,
    /// Link kind for the fitted family. Used by `predict_ci` and
    /// `predict_diff` to map between η and μ scales.
    pub link_kind: LinkKind,
    /// Family-specific shape parameters fit jointly with β / ρ. Empty
    /// for families that don't profile shape parameters (Gaussian,
    /// Bernoulli, Poisson, …). For Ocat: the `R − 2` log-gap thresholds
    /// (`θ_j`) that map to the `R + 1` category boundaries. For TDist:
    /// `[log_nu, log_sigma2]`. Older serialized models default to an
    /// empty array (via `#[serde(default)]`).
    #[cfg_attr(feature = "persistence", serde(default))]
    pub shape_params: Array1<f64>,
    /// Diagnostic counters captured during the fit. See
    /// [`crate::stats::FitStatsSnapshot`]. Always populated for new fits;
    /// `default()` (all zeros) for models deserialised from pre-0.5
    /// snapshots.
    #[cfg_attr(feature = "persistence", serde(default))]
    pub stats: crate::stats::FitStatsSnapshot,
}

impl FittedGam {
    /// Coefficients the design could not identify, as indices into `beta`.
    /// They are fitted as exactly 0 with zero variance (mgcv's convention), so
    /// they are read back from that rather than stored.
    pub fn aliased_coefficients(&self) -> Vec<usize> {
        (0..self.beta.len())
            .filter(|&i| self.beta[i] == 0.0 && self.vcov[[i, i]] == 0.0)
            .collect()
    }

    /// Predict on new x. `x_new` has shape `(n_new, n_input_dims)`;
    /// today every Predictor consumes a single column (epic 94a).
    /// Delegates to the [`Predictor`] for design reconstruction.
    /// Returns η (linear predictor) of length `n_new`; for
    /// response-scale predictions, apply [`LinkKind::inverse`]
    /// elementwise.
    pub fn predict(&self, x_new: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self.predictor.design(x_new)?.dot(&self.beta))
    }

    /// Predict `∂μ̂/∂x` at `x_new`. Identity-link Phase-0 semantics:
    /// `B'(x_new) · β_smooth`. The intercept's contribution is zero, so
    /// we slice off the leading column of the design derivative.
    pub fn predict_deriv(&self, x_new: ArrayView2<f64>) -> Result<Array1<f64>> {
        let d1 = self.predictor.design_deriv(x_new, 0)?;
        let beta_smooth = self.beta.slice(ndarray::s![1..]);
        Ok(d1.slice(ndarray::s![.., 1..]).dot(&beta_smooth))
    }

    /// Posterior covariance `σ̂² · A⁻¹` of β̂ on the fit basis. Returns a
    /// view of the cached field set at fit-time.
    pub fn vcov(&self) -> &Array2<f64> {
        &self.vcov
    }

    /// Wald-style pointwise confidence interval for predictions at new
    /// x. Returns `(mean, lo, hi)` as three 1-D arrays.
    ///
    /// - On the `Link` scale: `(η̂, η̂ − z·SE, η̂ + z·SE)` with
    ///   `SE_i = sqrt(x_iᵀ · vcov · x_i)` and `z = Φ⁻¹((1+level)/2)`.
    /// - On the `Response` scale: endpoints are inverse-linked. Note
    ///   that the SE itself is computed on the η scale (Wald is exact
    ///   only in η); response-scale `lo`/`hi` are therefore
    ///   `g⁻¹(η ± z·SE)` rather than a symmetric ±SE around `μ̂`.
    ///   This matches mgcv's convention.
    ///
    /// `level` is the central probability mass — e.g. 0.95 → ±1.96 SE.
    pub fn predict_ci(
        &self,
        x_new: ArrayView2<f64>,
        level: f64,
        scale: PredictScale,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        if !(0.0 < level && level < 1.0) {
            return Err(GamrsError::InvalidParameter(format!(
                "level must be in (0, 1); got level={level}"
            )));
        }
        let z = normal_quantile((1.0 + level) / 2.0);
        let design = self.predictor.design(x_new)?;
        let n_new = design.nrows();
        let eta = design.dot(&self.beta);

        // SE on the η scale: SE_i = sqrt(x_iᵀ · vcov · x_i).
        let mut se = Array1::<f64>::zeros(n_new);
        for i in 0..n_new {
            let row = design.row(i);
            let v = self.vcov.dot(&row.to_owned());
            let var_i: f64 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            se[i] = var_i.max(0.0).sqrt();
        }

        let mut lo = Array1::<f64>::zeros(n_new);
        let mut hi = Array1::<f64>::zeros(n_new);
        for i in 0..n_new {
            lo[i] = eta[i] - z * se[i];
            hi[i] = eta[i] + z * se[i];
        }

        match scale {
            PredictScale::Link => Ok((eta, lo, hi)),
            PredictScale::Response => {
                let mut mean_resp = Array1::<f64>::zeros(n_new);
                let mut lo_resp = Array1::<f64>::zeros(n_new);
                let mut hi_resp = Array1::<f64>::zeros(n_new);
                for i in 0..n_new {
                    mean_resp[i] = self.link_kind.inverse(eta[i]);
                    lo_resp[i] = self.link_kind.inverse(lo[i]);
                    hi_resp[i] = self.link_kind.inverse(hi[i]);
                }
                Ok((mean_resp, lo_resp, hi_resp))
            }
        }
    }

    /// Wald CI for the contrast `Δ = X_a·β̂ − X_b·β̂` on the η scale.
    /// Returns `(diff, lo, hi)` as three 1-D arrays of length
    /// `max(rows(X_a), rows(X_b))` (broadcasting when one is a single
    /// row).
    ///
    /// `var(Δ) = (X_a − X_b) · vcov · (X_a − X_b)ᵀ`; only the diagonal
    /// is used for pointwise CIs.
    pub fn predict_diff(
        &self,
        x_a: ArrayView2<f64>,
        x_b: ArrayView2<f64>,
        level: f64,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        if !(0.0 < level && level < 1.0) {
            return Err(GamrsError::InvalidParameter(format!(
                "level must be in (0, 1); got level={level}"
            )));
        }
        let design_a = self.predictor.design(x_a)?;
        let design_b = self.predictor.design(x_b)?;

        let (n_a, p_a) = (design_a.nrows(), design_a.ncols());
        let (n_b, p_b) = (design_b.nrows(), design_b.ncols());
        if p_a != p_b {
            return Err(GamrsError::InvalidParameter(format!(
                "predict_diff: design width mismatch (a={p_a}, b={p_b})"
            )));
        }

        let n_out = if n_a == n_b {
            n_a
        } else if n_a == 1 {
            n_b
        } else if n_b == 1 {
            n_a
        } else {
            return Err(GamrsError::InvalidParameter(format!(
                "predict_diff: row counts must match or one must be 1 (broadcast); got a={n_a}, b={n_b}"
            )));
        };

        let z = normal_quantile((1.0 + level) / 2.0);

        let mut diff = Array1::<f64>::zeros(n_out);
        let mut lo = Array1::<f64>::zeros(n_out);
        let mut hi = Array1::<f64>::zeros(n_out);
        let mut delta_row = Array1::<f64>::zeros(p_a);
        for i in 0..n_out {
            let ra = design_a.row(if n_a == 1 { 0 } else { i });
            let rb = design_b.row(if n_b == 1 { 0 } else { i });
            for j in 0..p_a {
                delta_row[j] = ra[j] - rb[j];
            }
            let d: f64 = delta_row
                .iter()
                .zip(self.beta.iter())
                .map(|(a, b)| a * b)
                .sum();
            let v = self.vcov.dot(&delta_row);
            let var_i: f64 = delta_row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let se = var_i.max(0.0).sqrt();
            diff[i] = d;
            lo[i] = d - z * se;
            hi[i] = d + z * se;
        }
        Ok((diff, lo, hi))
    }
}

/// Inverse CDF of the standard normal, used to convert a confidence
/// level to a z-score. Acklam's algorithm — accurate to ~1e-9 across
/// the central 0.99 of the distribution, plenty for Wald CIs (whose
/// approximation error dominates anyway).
///
/// Kept local (no extra crate dep) — gamrs intentionally avoids a stats
/// crate at this layer. Validates `0 < p < 1`; panics on out-of-range.
fn normal_quantile(p: f64) -> f64 {
    assert!(
        0.0 < p && p < 1.0,
        "normal_quantile: p must be in (0,1), got {p}"
    );
    // Acklam's algorithm.
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// =============================================================================
// Shared validation helpers — used by every per-family impl.
// =============================================================================

/// Fit `prep` with its unidentifiable coefficients removed, then put them back
/// as exact zeros with zero variance — mgcv's treatment of a rank-deficient
/// design. See [`crate::design::identifiability`].
pub(crate) fn fit_identifiable(
    prep: crate::design::PreparedDesign,
    prior_weights: Option<ArrayView1<f64>>,
    fit: impl FnOnce(crate::design::PreparedDesign) -> Result<FittedGam>,
) -> Result<FittedGam> {
    use crate::design::identifiability::{
        aliased_columns, with_zero_coefficients, with_zero_variance, without_columns,
    };
    let dropped = aliased_columns(&prep.x_design, &prep.s_list, prior_weights)?;
    if dropped.is_empty() {
        return fit(prep);
    }
    let mut fitted = fit(without_columns(prep, &dropped)?)?;
    fitted.beta = with_zero_coefficients(fitted.beta.view(), &dropped);
    fitted.vcov = with_zero_variance(&fitted.vcov, &dropped);
    Ok(fitted)
}

/// Check `x.nrows() == y.len()` and (optionally) `weights.len() == x.nrows()`.
pub(crate) fn check_lengths(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<()> {
    let n = x.nrows();
    if n != y.len() {
        return Err(GamrsError::InvalidParameter(format!(
            "x and y must have the same number of rows; got x.nrows()={}, y.len()={}",
            n,
            y.len()
        )));
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(GamrsError::InvalidParameter(format!(
                "prior_weights length must match x rows; got weights.len()={}, x.nrows()={}",
                w.len(),
                n
            )));
        }
    }
    Ok(())
}

/// Reject negative entries in `y`. Used by count families (Poisson,
/// QuasiPoisson, NegBin, Tweedie).
pub(crate) fn check_y_nonneg(y: ArrayView1<f64>, family_name: &str) -> Result<()> {
    for (i, &yi) in y.iter().enumerate() {
        if yi < 0.0 {
            return Err(GamrsError::InvalidParameter(format!(
                "{family_name} requires y ≥ 0; got y={yi} at row {i}"
            )));
        }
    }
    Ok(())
}

/// Reject non-positive entries in `y`. Used by strictly-positive continuous
/// families (Gamma, InverseGaussian).
pub(crate) fn check_y_positive(y: ArrayView1<f64>, family_name: &str) -> Result<()> {
    for (i, &yi) in y.iter().enumerate() {
        if yi <= 0.0 {
            return Err(GamrsError::InvalidParameter(format!(
                "{family_name} requires y > 0; got y={yi} at row {i}"
            )));
        }
    }
    Ok(())
}

/// Reject `y ∉ [0, 1]`. Used by Bernoulli + QuasiBinomial.
pub(crate) fn check_y_in_unit(y: ArrayView1<f64>, family_name: &str) -> Result<()> {
    for (i, &yi) in y.iter().enumerate() {
        if !(0.0..=1.0).contains(&yi) {
            return Err(GamrsError::InvalidParameter(format!(
                "{family_name} requires y in [0, 1]; got y={yi} at row {i}"
            )));
        }
    }
    Ok(())
}

/// Build `vcov = scale · A⁻¹` from a converged `GaussianInnerFit`. The
/// per-family caller passes its own `scale` (1.0 for fixed-φ, the
/// Pearson / mgcv-two-sigma φ̂ for profiled families). Used by every
/// `FittedGam` construction site so the vcov is populated once at
/// fit-time and downstream `predict_ci` / `predict_diff` calls are
/// cheap.
/// Coefficient covariance — mgcv's `Vp`.
///
/// From the **Fisher** pair where the family has one, for the same reason edf
/// is (see [`compute_edf`] and `Loss::expected_curvature_weights`): mgcv builds
/// `rV` from the `sqrt(wf)` re-factorisation at `gdi.c:2266-2288`, and
/// `Vp = rV·rVᵀ·scale`.
///
/// Verified in R against mgcv 1.9.4 on `1d_scat_saturated_basis_n620_k5_cr`:
/// the Fisher inverse matches `m$Vp` to **7.8e-15** max relative, while the
/// observed inverse — what this used unconditionally — is **2.2e-2** out
/// (`Vp[1,1]` 1.716070e+06 vs 1.713119e+06). Every standard error, confidence
/// interval and `predict(se = TRUE)` on a scat fit came off the wrong matrix.
pub(crate) fn compute_vcov<S: crate::inner::LinearSolver>(
    fit: &crate::inner::GaussianInnerFit<S>,
    scale: f64,
) -> ndarray::Array2<f64> {
    if let Some((f, _)) = fit.fisher.as_ref() {
        // The family's dispersion is ALREADY inside the Fisher weight — for
        // scat, `W_f = (ν+1)/((ν+3)σ²)` — so `A_f⁻¹` is already ∝ σ² and the
        // caller's `scale` must NOT be applied again. This is why mgcv reports
        // `scale = 1` for scat: `Vp = rV·rVᵀ·scale` with the dispersion carried
        // by `wf`, not by the multiplier.
        //
        // gamrs's scat driver reports `scale = σ²` (a deliberate divergence —
        // users want it), and `fit_shape_aware` passes that same value here. So
        // applying it would double-count: measured on
        // `1d_scat_saturated_basis_n620_k5_cr`, σ² = 0.686405 and the reported
        // vcov came out exactly `(1 − 0.686405) = 31.4%` away from mgcv's `Vp`.
        return S::invert(f);
    }
    let mut v = fit.a_inv();
    // Skip the multiplication if scale==1 — saves a full p² pass on the
    // fixed-φ families which are the majority.
    if (scale - 1.0).abs() > 0.0 {
        v.mapv_inplace(|x| x * scale);
    }
    v
}

/// EDF = `tr(H⁻¹ · X'WX)` using converged working-weights `w`. The trace
/// goes through `GaussianInnerFit::trace_a_inv` so it picks up whichever
/// `LinearSolver` backend the fit was produced with — closes audit
/// finding #4 (was duplicated as `trace_a_inv_s` here + `trace_solve` in
/// the score module + the elementwise pattern inline).
/// `edf = tr(A⁻¹·X'WX)`.
///
/// `W` MUST be the weights `fit.a_factor` was built from, not the IRLS working
/// weights — for an observed→expected switch family those differ on the outlier
/// rows, and mixing them inflates the trace (`A⁻¹` from the smaller observed
/// matrix against a larger `X'WX`), overshooting edf. Callers should pass
/// `fit.a_weights`; the parameter is kept explicit because the ocat path feeds
/// a ridged factor whose weights are its own.
pub(crate) fn compute_edf<S: crate::inner::LinearSolver>(
    x_design: &ndarray::Array2<f64>,
    working_weights: &Array1<f64>,
    fit: &crate::inner::GaussianInnerFit<S>,
) -> f64 {
    // mgcv computes edf from the FISHER pair, not from whatever factor the
    // score differentiates: `gdi.c:2260-2290` re-QRs `[sqrt(wf)X; E]` after the
    // observed-weight work is done, precisely so edf is "computed safely".
    // Where the family exposes that weight, use it — BOTH sides from `W_f`, its
    // own factorisation. Verified against mgcv 1.9.4 on
    // `1d_scat_saturated_basis_n620_k5_cr`: this reproduces mgcv's reported
    // 2.3879040656710795 to 4.5e-14, where pairing the observed weight with the
    // observed factor gives 2.3807272805 (7.2e-3 out, because 23 of 620 rows
    // have ½·D_μμ ≤ 0).
    let (weights, factor) = match fit.fisher.as_ref() {
        Some((f, w_f)) => (w_f, Some(f)),
        None => (working_weights, None),
    };
    let p = x_design.ncols();
    let n = x_design.nrows();
    let mut wxt = ndarray::Array2::<f64>::zeros((p, n));
    for j in 0..n {
        for i in 0..p {
            wxt[[i, j]] = x_design[[j, i]] * weights[j];
        }
    }
    let xtwx = wxt.dot(x_design);
    match factor {
        Some(f) => S::trace_a_inv(f, xtwx.view()),
        None => fit.trace_a_inv(xtwx.view()),
    }
}

/// Per-term EDF for a multi-smooth fit. Uses the identity
/// `edf_per_term[j] = dim(S_j) - λ_j · tr(H⁻¹ S_j)` where `dim(S_j)` is
/// the column-block size of term j (the rank+nullspace of its embedded
/// penalty restricted to its block). Sums to `edf_total - 1` modulo
/// per-term null spaces that contribute to the intercept's dof of 1.
///
/// For the single-smooth path with `s_list.len() == 1`,
/// `edf_per_term[0] = edf_total - 1` (the intercept's fixed dof).
pub(crate) fn compute_edf_per_term<S: crate::inner::LinearSolver>(
    s_list: &[ndarray::Array2<f64>],
    rho: &Array1<f64>,
    p_design: usize,
    fit: &crate::inner::GaussianInnerFit<S>,
) -> Array1<f64> {
    let n_terms = s_list.len();
    debug_assert_eq!(n_terms, rho.len());
    let mut edf_j = Array1::<f64>::zeros(n_terms);
    // `dim_j` = number of columns the term occupies in the embedded
    // penalty (i.e. count of rows with any nonzero entry in S_j). For
    // block-diagonal penalty layout each term's block has full coverage
    // over its column range.
    for j in 0..n_terms {
        let s_j = &s_list[j];
        let lambda_j = rho[j].exp();
        let mut dim_j = 0usize;
        for r in 0..p_design {
            let mut row_nonzero = false;
            for c in 0..p_design {
                if s_j[[r, c]] != 0.0 {
                    row_nonzero = true;
                    break;
                }
            }
            if row_nonzero {
                dim_j += 1;
            }
        }
        let tr_hinv_s_j = fit.trace_a_inv(s_j.view());
        edf_j[j] = (dim_j as f64) - lambda_j * tr_hinv_s_j;
    }
    edf_j
}
