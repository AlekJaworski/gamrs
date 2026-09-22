//! Layer 3 — inner solver: β̂(θ) given fixed smoothing parameters θ.
//!
//! Split layout (algorithm-named):
//! - `mod.rs`        — shared `GaussianInnerFit`, plumbing helpers, public
//!                     re-exports of solvers below.
//! - `linalg`        — `LinearSolver` trait + `CholeskySolver` /
//!                     `LuSolver` backends. Every inner solver is generic
//!                     over `S: LinearSolver` with default `CholeskySolver`.
//! - `closed_form`   — `GaussianClosedFormInner`, `gaussian_inner_solve` —
//!                     one Cholesky, no IRLS iteration (identity link +
//!                     constant weights).
//! - `pirls`         — `PirlsInner` — penalised IRLS for canonical-link GLM
//!                     exponential families.
//! - `gam_fit5`      — `OcatInner` — gam.fit5-style joint β + working-weights
//!                     fit for the ocat extended family.
//! - `armijo`        — `ArmijoElfInner` — Armijo-backtracking Newton for
//!                     ELF / quantile families.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{Cholesky, UPLO};

use crate::error::{GamrsError, Result};

pub mod armijo;
pub mod closed_form;
pub mod gam_fit5;
pub mod linalg;
pub mod pirls;

pub use armijo::{ArmijoElfInner, ArmijoElfOpts};
pub use closed_form::{gaussian_inner_solve, GaussianClosedFormInner};
pub use gam_fit5::OcatInner;
pub use linalg::{
    chol_back_solve, chol_forward_solve, factor_and_solve, CholeskySolver, LinearSolver, LuSolver,
};
/// Diagnostic re-exports for the observed-curvature criterion's PD-fallback
/// counters. These exist alongside the migration switch and go with it.
pub use pirls::observed_pd_counts as pirls_observed_pd_counts;
pub use pirls::reset_observed_pd_counts as pirls_reset_observed_pd_counts;
pub use pirls::{PirlsInner, PirlsOpts};

/// Result of any inner-solver fit — Gaussian or PIRLS-iterative.
///
/// The same struct serves both paths so `ScoreDerivatives` impls can be
/// `InnerSolver`-generic without an associated-type explosion. PIRLS impls
/// fill `eta` / `mu` / `working_weights` at convergence; Gaussian's one-
/// solve path fills `mu = X·β` and `working_weights = prior_weights ∨ ones`.
///
/// Parameterised over `S: LinearSolver` (default `CholeskySolver`). The
/// stored `a_factor: S::Factorization` is the converged factorisation of
/// `A = X' diag(W) X + λS` — the score-derivative layer goes through
/// [`GaussianInnerFit::log_det_a`] / [`GaussianInnerFit::trace_a_inv`]
/// rather than touching the factor directly. Switching the backend
/// (`gamrs::fit::<_, _, _, LuSolver>(...)`) flips every consumer at the
/// type level — no string keys, no runtime branching.
pub struct GaussianInnerFit<S: LinearSolver = CholeskySolver> {
    pub beta: Array1<f64>,
    /// Linear predictor `η = X·β` at convergence.
    pub eta: Array1<f64>,
    /// Fitted mean `μ = g⁻¹(η)` at convergence.
    pub mu: Array1<f64>,
    /// Working weights at convergence — `W_ii = 1/(V(μ_i) · g'(μ_i)²)` for
    /// PIRLS, or prior weights for the Gaussian closed-form path.
    pub working_weights: Array1<f64>,
    /// Working response `z = η + (y - μ) · g'(μ)` at convergence. Used by
    /// the closed-form score's `Dp/(2σ²) + log|H|/2` formula since
    /// `H = X' diag(W) X + λS`.
    pub working_response: Array1<f64>,
    /// **Unpenalized deviance** `Σ D_i(y_i, μ_i)`. For Gaussian + identity
    /// this equals `Σ w_i (y_i - μ_i)²`; for other families it's the
    /// family-specific GLM deviance.
    pub deviance: f64,
    /// For backward compatibility with the Gaussian-only score code path
    /// (this is just `deviance` for Gaussian; equal for the closed-form
    /// path because identity link makes μ = X·β = η and z = η).
    pub rss: f64,
    pub n: usize,
    pub p: usize,
    pub iterations: usize,
    pub converged: bool,
    /// Factorisation of `A = X' diag(W) X + λS` at convergence, kept in
    /// the backend-native form (lower Cholesky factor for the
    /// `CholeskySolver` default, LU pivots for `LuSolver`). Reused by the
    /// score-derivative layer through [`GaussianInnerFit::log_det_a`] /
    /// [`GaussianInnerFit::trace_a_inv`] — never re-form `A` explicitly.
    pub a_factor: S::Factorization,
    /// The per-row weights `a_factor` was actually built from.
    ///
    /// `a_factor` and this vector are ONE pair and must never be mixed with a
    /// different `W`. `compute_edf` forms `tr(A⁻¹·X'WX)`, so pairing this
    /// factor with `working_weights` instead — which for an observed→expected
    /// switch family carries the substituted positive curvature — inflates the
    /// trace and overshoots edf. Same class of bug as a score whose gradient
    /// differentiates a different matrix than its value: if two quantities
    /// have to agree, carry them together rather than re-deriving one.
    ///
    /// Equal to `working_weights` on every path that does not substitute.
    pub a_weights: Array1<f64>,
    /// The **Fisher** pair: `(factorisation of X'diag(W_f)X + λS, W_f)`, where
    /// `W_f` is `Loss::expected_curvature_weights`.
    ///
    /// A SECOND factorisation, deliberately. `a_factor` is what the score
    /// differentiates (observed curvature); this is what edf and the
    /// coefficient covariance are computed from. mgcv keeps both for exactly
    /// this reason (`gdi.c:2260-2290` re-QRs with `sqrt(wf)` after the observed
    /// work is done). One factor cannot serve both jobs.
    ///
    /// `None` when the family supplies no separate expected curvature, in which
    /// case edf and vcov fall back to `a_factor` / `a_weights` as before.
    pub fisher: Option<(S::Factorization, Array1<f64>)>,
    /// Optional `log|H|` override for the REML score body. Defaults to
    /// `None`, which makes the score body read `2·Σ log L_ii` off
    /// `a_factor` (the Fisher-W A). PIRLS solvers fill this with the
    /// Newton-W `Σ log|λ_i|` via `eigh` when the loss declared
    /// `use_newton_irls() = true` — mirroring v0.x
    /// `src/reml/mod.rs:436-459`, which rebuilds `A_score = X'·W_newton·X
    /// + λS` and uses `log_abs_det_symmetric` to handle the indefinite
    /// spectrum that ~43% of negative-α rows induce on InverseGaussian +
    /// log. `tr(H⁻¹S)` keeps using the Fisher H per v0.x's
    /// `system.tr_a` convention (the score's `(n − tr_a)` denominator
    /// expects Fisher).
    pub log_det_h_override: Option<f64>,
    /// Optional Tk·KK' gradient inputs for non-canonical-link families.
    /// When `Some`, the REML score's ρ-gradient adds the
    /// `Σᵢ a1[i] · η₁[i] · sign(w[i]) · lev_uw[i]` term that v0.x
    /// `src/reml/mod.rs::reml_gradient_mgcv_exact_ift_inner_at_beta`
    /// computes for InverseGaussian / Binomial / QuasiBinomial. The
    /// derivative `η₁ = X·b1 = -λ·X·A⁻¹·S·β` and the score-side
    /// `lev_uw` solve are deferred to the score body (it already owns
    /// `A⁻¹` via `a_factor`); PIRLS supplies the link/variance-
    /// derivative bits that depend on the family.
    ///
    /// Layout:
    /// - `a1`: length n. v0.x `gdi.c:2535/2556` — Newton-mode value
    ///   `a1[i] = w[i]·(α₁ - V'/V - 2·g''/g')/g'(μ)` (Fisher fallback
    ///   collapses to `-w[i]·(V'/V + 2·g''/g')/g'(μ)` when α≤0).
    /// - `lev_uw`: length n. Unweighted leverage `x_iᵀ A⁻¹ x_i`.
    /// - `x_for_eta1`: the design matrix (cloned reference) so the score
    ///   can build `η₁ = X · b1` without re-importing it. gamrs's PIRLS
    ///   already owns `self.x_design`; this is just a handle so the
    ///   score doesn't need a parallel field.
    pub tk_kkt_inputs: Option<TkKKTInputs>,
    /// Per-observation `∂W_ii/∂η_i` at convergence, or `None` when the
    /// working weights are constant in η (Gaussian closed-form path).
    ///
    /// The analytic outer-Newton Hessian needs this because the gradient's
    /// `½·λ_j·tr(A⁻¹S_j)` term has `A = X'WX + ΣλS` with `W = W(η(β(ρ)))`
    /// for GLM / quantile families: `∂tr(A⁻¹S_j)/∂ρ_i` therefore carries a
    /// `−tr(A⁻¹·X'(∂W/∂ρ_i)X·A⁻¹·S_j)` piece on top of the penalty part,
    /// where `∂W/∂ρ_i = diag((∂W/∂η)·(X·dβ/dρ_i))`. Without it the analytic
    /// Hessian is the fixed-W (Fisher) Hessian, which disagrees with the
    /// finite-difference Hessian of the same gradient by O(1e-3) for PIRLS
    /// families. Gaussian leaves it `None` (W constant ⇒ term vanishes).
    pub dw_deta: Option<Array1<f64>>,
    /// Design matrix `X` (cloned), populated only when `dw_deta` is `Some`
    /// — the analytic Hessian's W-chain term needs `X·dβ/dρ` and the
    /// leverage-like `(X·A⁻¹S_jA⁻¹·X')_{ii}`. Gaussian leaves it `None`.
    pub x_design: Option<Array2<f64>>,
}

impl<S: LinearSolver> GaussianInnerFit<S> {
    /// `log|A|` from the stored factorisation. Cholesky path: `2·Σ log L_ii`.
    /// LU path: `log|det A|` via mantissa+exponent decomposition.
    #[inline]
    pub fn log_det_a(&self) -> f64 {
        S::logdet(&self.a_factor)
    }

    /// `tr(A⁻¹·M)` via the backend's elementwise pattern (matches v0.x's
    /// `src/reml/mod.rs:914-918` iteration order — closes audit finding
    /// #4 by collapsing the duplicate `trace_solve` impls).
    #[inline]
    pub fn trace_a_inv(&self, m: ArrayView2<f64>) -> f64 {
        S::trace_a_inv(&self.a_factor, m)
    }

    /// Solve `A·x = b` using the stored factorisation. Used by the score
    /// body for `b1 = -λ · A⁻¹ · S · β` and for the leverage solves.
    #[inline]
    pub fn solve_a(&self, b: ArrayView1<f64>) -> Array1<f64> {
        S::solve(&self.a_factor, b)
    }

    /// Form `A⁻¹` explicitly. Used by the Tk·KK' pipeline when many
    /// solves against the same A are required (`A⁻¹·x_i` for every row).
    #[inline]
    pub fn a_inv(&self) -> Array2<f64> {
        S::invert(&self.a_factor)
    }
}

/// The score body assembles the Tk·KK' gradient contribution
/// `tk_kkt[k] = Σᵢ a1[i] · η₁[i,k] · lev_uw[i]` from these inputs (the
/// `sign(w)` factor in v0.x cancels — `gdi.c:856` derivation).
/// Built by `PirlsInner` at convergence for non-canonical families;
/// `None` for the canonical-link Fisher path (term identically zero by
/// envelope theorem on the W-β chain).
///
/// **All quantities use the Newton-W A matrix** (`X' diag(w_newton) X +
/// λS`, computed via eigendecomposition because it can be indefinite for
/// ~43% negative-α rows on InverseGaussian + log). v0.x's
/// `reml_gradient_mgcv_exact_ift_newton_at_beta` (`src/reml/mod.rs:2322-
/// 2463`) does the same — using a mix of Newton- and Fisher- A would
/// produce a gradient that doesn't differentiate the score's Newton
/// `log|H|`.
///
/// 1-D ρ (Phase-8 IG): `eta1` is `∂η/∂ρ = -λ · X · A_newton⁻¹ · S · β`.
#[derive(Clone)]
pub struct TkKKTInputs {
    /// Newton-mode IFT weight derivative — mgcv `gdi.c:2556`. Uses
    /// `w_newton[i] = wf · α` (no Fisher fallback).
    pub a1: ndarray::Array1<f64>,
    /// Unweighted leverage `x_iᵀ A_newton⁻¹ x_i`.
    pub lev_uw: ndarray::Array1<f64>,
    /// Per-term `η₁_k = ∂η/∂ρ_k = -λ_k · X · A_newton⁻¹ · S_k · β` (each
    /// length n). Length = m smooths. Single-smooth families just have
    /// `eta1_per_term.len() == 1`. Port of mgcv_rust
    /// `reml_gradient_mgcv_exact_ift_newton_at_beta` (multi-smooth
    /// `eta1[:, k]` columns at `src/reml/mod.rs:2401-2402`).
    pub eta1_per_term: Vec<ndarray::Array1<f64>>,
    /// Per-term `tr(A_newton⁻¹ · S_k)` — override for the score's
    /// `λ_k·tr(H⁻¹S_k)` term that must use the same Newton A as the
    /// Tk·KK' machinery. Without this the gradient's `tr` and `tk_kkt`
    /// would derive from different A matrices.
    pub tr_a_newton_inv_s_per_term: Vec<f64>,
    /// `A_newton⁻¹` — the dense inverse of the Newton observed-info A
    /// matrix at converged β. The shape-aware IFT analytic θ-gradient
    /// uses this (NOT `fit.a_factor`, which is Fisher-A) so it
    /// differentiates the SAME `log|H|` that the score's Newton override
    /// computes — mgcv `gam.fit4.r:gdi2` consistency. Materialised once
    /// by `compute_tk_kkt_inputs` via the same eigendecomposition that
    /// builds `a1` / `lev_uw`.
    pub a_newton_inv: ndarray::Array2<f64>,
    /// Unused now (kept for backward compat with the field name); will
    /// drop on v0.3 alongside any other `sign(w)` traces.
    pub working_weights_sign: ndarray::Array1<f64>,
}

// =============================================================================
// Shared IRLS plumbing — used by every iterative inner solver below.
// =============================================================================

/// `X' diag(w)` as a `(p, n)` matrix, written without forming `diag(w)`.
pub(crate) fn weighted_xt(x_design: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    weighted_xt_view(x_design.view(), w.view())
}

/// View-accepting variant — used at hot `xtwx_xtwy` callers to avoid
/// `to_owned()` clones (160 MB per PIRLS iter at n=1M, p=20).
pub(crate) fn weighted_xt_view(x_design: ArrayView2<f64>, w: ArrayView1<f64>) -> Array2<f64> {
    let n = x_design.nrows();
    let p = x_design.ncols();
    let mut xtw = Array2::<f64>::zeros((p, n));
    for j in 0..n {
        let wj = w[j];
        for i in 0..p {
            xtw[[i, j]] = x_design[[j, i]] * wj;
        }
    }
    xtw
}

/// `β' S β`. Quadratic-penalty contribution, multiplied by λ to obtain the
/// `λ β'Sβ` term in `pdev = Σw·D + λ·β'Sβ`.
pub(crate) fn beta_sbeta(s: &Array2<f64>, beta: &Array1<f64>) -> f64 {
    let p = beta.len();
    let mut acc = 0.0_f64;
    for i in 0..p {
        let mut row = 0.0;
        for j in 0..p {
            row += s[[i, j]] * beta[j];
        }
        acc += beta[i] * row;
    }
    acc
}

/// Add `λ · S` to `xtwx` in place (penalty assembly).
#[inline]
pub(crate) fn add_penalty(a: &mut Array2<f64>, s: &Array2<f64>, lambda: f64) {
    let p = a.nrows();
    for i in 0..p {
        for j in 0..p {
            a[[i, j]] += lambda * s[[i, j]];
        }
    }
}

/// Add `1e-7 · max(|A_ii|, 1)` to the diagonal of `a` and return its lower
/// Cholesky factor. Mirrors v0.x's `pirls_ridge_scale` defensive ridge
/// used when working weights collapse (ELF saturation, ocat near-boundary).
///
/// **Kept Cholesky-specific** because the call sites (ELF Armijo, ocat,
/// Quantile warm-start) use the lower factor directly via
/// `chol_forward_solve` / `chol_back_solve`. Switching them to the generic
/// `LinearSolver::factorize(...)` path is a separate refactor — it would
/// require those inner solvers to also become S-generic, and they currently
/// only run with `CholeskySolver`. (Their `GaussianInnerFit<S>` factor is
/// still backend-correct: it's a fresh `S::factorize(a)` call at the end.)
pub(crate) fn cholesky_with_safety_ridge(mut a: Array2<f64>, label: &str) -> Result<Array2<f64>> {
    let p = a.nrows();
    let mut max_diag = 1.0_f64;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge = 1e-7 * max_diag;
    for i in 0..p {
        a[[i, i]] += ridge;
    }
    a.cholesky(UPLO::Lower)
        .map_err(|e| GamrsError::SingularSystem(format!("{label} Cholesky: {e}")))
}

/// Generic mgcv-style step-halving toward `beta_old`. Re-evaluates `(eta,
/// dev, pdev)` after each halving via `recompute`; declares a needs-halve
/// when `pdev` is non-finite, `is_invalid(η)` fires, or — only on
/// `iter > 0` — the penalised deviance diverges by more than `div_thresh
/// = 10·(0.1+|pdev_old|)·√ε` (the mgcv `gam.fit3.r:382-441` guard).
///
/// `recompute(beta) -> (eta, dev, pdev, mu_opt)`. `mu_opt` lets PIRLS
/// report `μ = inverse_link(η)` (validated against `Loss::deviance_per_obs`
/// finiteness); ocat passes `None` because μ = η.
///
/// Returns `(beta_accepted, eta, dev, pdev, mu_opt, accepted_flag)`. When
/// `accepted_flag` is false, 100 halvings were exhausted and the caller
/// should bail with the last successful state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn halve_until_valid<F, V>(
    mut beta_try: Array1<f64>,
    beta_old: &Array1<f64>,
    mut eta_try: Array1<f64>,
    mut dev_try: f64,
    mut pdev_try: f64,
    mut mu_try: Option<Array1<f64>>,
    pdev_old: f64,
    iter_one: bool,
    mut recompute: F,
    is_invalid: V,
) -> (
    Array1<f64>,
    Array1<f64>,
    f64,
    f64,
    Option<Array1<f64>>,
    bool,
)
where
    F: FnMut(&Array1<f64>) -> (Array1<f64>, f64, f64, Option<Array1<f64>>),
    V: Fn(&Array1<f64>, Option<&Array1<f64>>) -> bool,
{
    let p = beta_try.len();
    let div_thresh = 10.0 * (0.1 + pdev_old.abs()) * f64::EPSILON.sqrt();
    let needs_halve = |pdev_t: f64, eta_t: &Array1<f64>, mu_t: Option<&Array1<f64>>| -> bool {
        if !pdev_t.is_finite() {
            return true;
        }
        if is_invalid(eta_t, mu_t) {
            return true;
        }
        !iter_one && pdev_t - pdev_old > div_thresh
    };

    let mut halvings = 0usize;
    while needs_halve(pdev_try, &eta_try, mu_try.as_ref()) && halvings < 100 {
        for j in 0..p {
            beta_try[j] = 0.5 * (beta_try[j] + beta_old[j]);
        }
        let (e, d, pd, m) = recompute(&beta_try);
        eta_try = e;
        dev_try = d;
        pdev_try = pd;
        mu_try = m;
        halvings += 1;
    }
    let accepted = !needs_halve(pdev_try, &eta_try, mu_try.as_ref());
    (beta_try, eta_try, dev_try, pdev_try, mu_try, accepted)
}

/// Build `X'WX` and `X'Wy` once. Both are needed by the inner solver and
/// by the score's effective-dof / σ̂² computation.
pub(crate) fn xtwx_xtwy(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: Option<ArrayView1<f64>>,
) -> (Array2<f64>, Array1<f64>) {
    match w {
        Some(w) => {
            // Avoid to_owned() clones of x/w (combined ~168 MB at n=1M)
            // by routing the view-accepting weighted_xt directly.
            let wxt = weighted_xt_view(x, w);
            (wxt.dot(&x), wxt.dot(&y))
        }
        None => (x.t().dot(&x), x.t().dot(&y)),
    }
}
