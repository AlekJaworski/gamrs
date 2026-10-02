//! Penalised IRLS for canonical-link GLM exponential families.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::{GamrsError, Result};
use crate::family::Family;
use crate::traits::{InnerSolver, Link, Loss, VarianceFn};

use super::{
    add_penalty, beta_sbeta, factor_and_solve, halve_until_valid, weighted_xt, CholeskySolver,
    GaussianInnerFit, LinearSolver,
};

/// Vectorised Newton score weights `w_newton[i] = wf · α` at converged β.
/// No Fisher fallback — negative α stay negative. Port of mgcv_rust
/// `src/pirls/row_step.rs::compute_newton_score_weights` (line 117-127);
/// consumed by the lazy Newton log|H| / Tk·KK' helpers below.
pub(crate) fn newton_score_weights<L, K, V>(
    family: &Family<L, K, V>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    prior_w: &Array1<f64>,
) -> Array1<f64>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
{
    let n = y.len();
    let mut w_newton = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mu_i = mu[i];
        let var_i = family.variance.variance(mu_i).max(1e-300);
        let g_prime_mu = family.link.d_link_dmu(mu_i);
        let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
        let v_prime = family.variance.d_variance(mu_i);
        let v1n = v_prime / var_i;
        let g_double_prime = family.link.d2_link_dmu(mu_i);
        let g2n = g_double_prime / g_prime_mu;
        let c_resid = y[i] - mu_i;
        let alpha = 1.0 + c_resid * (v1n + g2n);
        w_newton[i] = prior_w[i] * wf * alpha;
    }
    w_newton
}

/// Lazy Newton log|H| at converged β. Standalone port of mgcv_rust
/// `src/reml/mod.rs:460-483`. Builds `A_score = X' diag(W_newton) X + λS`
/// then factors:
/// - **Cholesky first** (`2·Σ log L_ii`) — succeeds when α > 0 everywhere
///   (NegBin: always; IG + log: ~57% of fixtures). O(p³/3).
/// - **eigh fallback** (`Σ log|λᵢ|`) when Cholesky fails (indefinite A).
///   O(~3p³).
///
/// Returns `None` if neither path produces finite output (caller falls back
/// to the Fisher H's log|H|).
/// Counters for how often the observed penalised Hessian is usable.
///
/// The observed-curvature criterion is CONDITIONALLY defined: when
/// `X'diag(½D_μμ)X + λS` loses positive-definiteness the fit keeps the
/// working-weight factor instead, which means the objective changed under the
/// optimiser's feet. These count the two branches so that discontinuity is
/// measured rather than assumed — it has never been observed to fire.
///
/// Incremented at the `a_factor` rebuild, which is where the decision lives
/// now that the score reads `log|H|` straight off that factor.
pub(crate) static OBS_PD_OK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static OBS_PD_FALLBACK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `(pd_ok, pd_fallback)` since the last reset.
pub fn observed_pd_counts() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (OBS_PD_OK.load(Relaxed), OBS_PD_FALLBACK.load(Relaxed))
}

pub fn reset_observed_pd_counts() {
    use std::sync::atomic::Ordering::Relaxed;
    OBS_PD_OK.store(0, Relaxed);
    OBS_PD_FALLBACK.store(0, Relaxed);
}

/// The observed curvature the score should differentiate, or `None` to leave
/// the working-weight factor alone — the single seam that decides which matrix
/// `fit.a_factor` is, and therefore what `log|H|`, `tr(H⁻¹S)`, `a_inv`, the
/// gradient's `h_diag` and the Hessian all read. One place, so they cannot
/// disagree.
fn family_observed_score_weights<L, K, V>(
    family: &Family<L, K, V>,
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_w: &Array1<f64>,
) -> Option<Array1<f64>>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
{
    let w = family
        .loss
        .observed_curvature_weights(y.view(), eta.view(), Some(prior_w.view()))?;
    if w.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(w)
}

/// **The** `log|H|` the score differentiates. One definition of the precedence.
///
/// Newton-A (`use_newton_irls`) → the fit's own factor. There used to be a
/// third, leading branch that rebuilt the observed penalised Hessian here; it
/// went when `pirls.rs` started building `fit.a_factor` from the observed
/// curvature directly, which makes `fit.log_det_a()` that same number without
/// an O(n·p²) product and a Cholesky per score evaluation.
///
/// The precedence lives in ONE place because the scat Python diagnostic's copy
/// of it had drifted to a bare `fit.log_det_a()`, so a profile taken with it
/// measured a different function than the fitter optimised.
pub(crate) fn score_log_det_h<L, K, V, S>(
    family: &Family<L, K, V>,
    y: &Array1<f64>,
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    prior_w: &Array1<f64>,
    x_design: &Array2<f64>,
    s_total: &Array2<f64>,
    fit: &crate::inner::GaussianInnerFit<S>,
) -> f64
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: crate::inner::LinearSolver,
{
    let _ = eta;
    if family.loss.use_newton_irls() {
        lazy_newton_log_det_h(family, y, mu, prior_w, x_design, s_total)
            .unwrap_or_else(|| fit.log_det_a())
    } else {
        fit.log_det_a()
    }
}

/// Cheap precondition for [`score_log_det_h`]: when this is false the caller
/// must skip it entirely rather than pay to build `s_total` for nothing.
///
/// A family with `observed_curvature_weights` needs the observed `log|H|`; a
/// `use_newton_irls` family needs the Newton one. Everything else keeps the
/// fit's own factor and pays nothing.
pub(crate) fn score_log_det_h_applies<L: Loss + Clone>(loss: &L) -> bool {
    // NOT `has_observed_curvature()`. For such a family `pirls.rs` already
    // built `fit.a_factor` FROM the observed curvature, so `fit.log_det_a()` is
    // `log|H_obs|` — measured equal to a fresh `observed_log_det_h` to
    // 0.000e0 at every probe. Asking for it again re-does an O(n·p²) weighted
    // product plus a Cholesky, per score evaluation, to reproduce a number the
    // fit is holding. Only the Newton-A path needs a separate build.
    loss.use_newton_irls()
}

pub(crate) fn lazy_newton_log_det_h<L, K, V>(
    family: &Family<L, K, V>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    prior_w: &Array1<f64>,
    x_design: &Array2<f64>,
    s_total: &Array2<f64>,
) -> Option<f64>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
{
    use ndarray_linalg::{Cholesky, Eigh, UPLO};
    let n = x_design.nrows();
    let p = x_design.ncols();

    let w_newton = newton_score_weights(family, y, mu, prior_w);
    for &w in w_newton.iter() {
        if !w.is_finite() {
            return None;
        }
    }

    // Build A_score = X' diag(W_newton) X + λS (single combined penalty).
    // BLAS-accelerated form: WX = diag(w)·X (per-row scale), then X'·WX.
    // Mirrors mgcv_rust `src/reml/mod.rs:2366-2373`'s in-place row scaling
    // followed by `x.t().dot(&wx)`. Manual triple-loop was the dominant
    // O(n·p²) cost on the NegBin bench — replacing with BLAS .dot() drops
    // it by an order of magnitude.
    let mut wx = x_design.clone();
    for i in 0..n {
        let wi = w_newton[i];
        for j in 0..p {
            wx[[i, j]] *= wi;
        }
    }
    let mut a_score: Array2<f64> = x_design.t().dot(&wx);
    for j in 0..p {
        for l in 0..p {
            a_score[[j, l]] += s_total[[j, l]];
        }
    }
    // Symmetrise defensively before factor.
    for j in 0..p {
        for l in (j + 1)..p {
            let avg = 0.5 * (a_score[[j, l]] + a_score[[l, j]]);
            a_score[[j, l]] = avg;
            a_score[[l, j]] = avg;
        }
    }

    // PSD-fast path: Cholesky. Cheap log|H| = 2·Σ log L_ii.
    if let Ok(l) = a_score.cholesky(UPLO::Lower) {
        let mut log_det = 0.0_f64;
        for i in 0..p {
            let lii = l[[i, i]];
            if !lii.is_finite() || lii.abs() < 1e-300 {
                return None;
            }
            log_det += lii.ln();
        }
        return Some(2.0 * log_det);
    }

    // Indefinite-A fallback — eigh.
    let eigs = match a_score.eigh(UPLO::Lower) {
        Ok((eigs, _)) => eigs,
        Err(_) => return None,
    };
    let mut log_det = 0.0_f64;
    for e in eigs.iter() {
        let ae = e.abs();
        if ae < 1e-300 || !ae.is_finite() {
            return None;
        }
        log_det += ae.ln();
    }
    Some(log_det)
}

/// Lazy Tk·KK' / IFT inputs at converged β. Standalone port of mgcv_rust
/// `src/reml/mod.rs::reml_gradient_mgcv_exact_ift_newton_at_beta`
/// (`src/reml/mod.rs:2347-2487`) — builds `A_newton = X' diag(w_newton) X +
/// λS`, factors via Cholesky-first (eigh fallback), forms `A_newton⁻¹`,
/// then assembles `{a1, lev_uw, eta1_per_term, tr_a_newton_inv_s_per_term}`.
///
/// Returns `None` on factor failure (caller bails to no-Tk·KK' branch).
pub(crate) fn lazy_tk_kkt_inputs<L, K, V>(
    family: &Family<L, K, V>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    beta: &Array1<f64>,
    prior_w: &Array1<f64>,
    x_design: &Array2<f64>,
    s_list: &[Array2<f64>],
    s_total: &Array2<f64>,
    rho: &Array1<f64>,
) -> Option<super::TkKKTInputs>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
{
    use ndarray_linalg::{Cholesky, Eigh, UPLO};
    let n = x_design.nrows();
    let p = x_design.ncols();

    // Newton weights w_newton[i] = wf · α — NO prior_w factor (mgcv_rust's
    // `compute_newton_score_weights` uses pure family weights; prior_w is
    // applied separately in the score formula). Match v0.x's
    // `compute_tk_kkt_inputs` which used `wf · α` without prior_w.
    let mut w_newton = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mu_i = mu[i];
        let var_i = family.variance.variance(mu_i).max(1e-300);
        let g_prime_mu = family.link.d_link_dmu(mu_i);
        let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
        let v_prime = family.variance.d_variance(mu_i);
        let v1n = v_prime / var_i;
        let g_double_prime = family.link.d2_link_dmu(mu_i);
        let g2n = g_double_prime / g_prime_mu;
        let c_resid = y[i] - mu_i;
        let alpha = 1.0 + c_resid * (v1n + g2n);
        w_newton[i] = wf * alpha;
        if !w_newton[i].is_finite() {
            return None;
        }
    }

    // Build A_newton = X' diag(w_newton) X + λS. BLAS form (see
    // `lazy_newton_log_det_h` for the same pattern, mirroring mgcv_rust
    // `src/reml/mod.rs:2366-2373`).
    let mut wx = x_design.clone();
    for i in 0..n {
        let wi = w_newton[i];
        for j in 0..p {
            wx[[i, j]] *= wi;
        }
    }
    let mut a_newton: Array2<f64> = x_design.t().dot(&wx);
    for j in 0..p {
        for l in 0..p {
            a_newton[[j, l]] += s_total[[j, l]];
        }
    }
    // Symmetrise.
    for j in 0..p {
        for l in (j + 1)..p {
            let avg = 0.5 * (a_newton[[j, l]] + a_newton[[l, j]]);
            a_newton[[j, l]] = avg;
            a_newton[[l, j]] = avg;
        }
    }

    // A_newton⁻¹ — Cholesky-first (PSD common case for NegBin α>0); eigh
    // fallback for indefinite spectra (IG + log ~43% negative-α path).
    let a_inv: Array2<f64> = if let Ok(l) = a_newton.cholesky(UPLO::Lower) {
        // Materialise inverse via column-wise back-solve.
        let mut a_inv = Array2::<f64>::zeros((p, p));
        for col in 0..p {
            let mut e_j = Array1::<f64>::zeros(p);
            e_j[col] = 1.0;
            let z = super::chol_forward_solve(&l, e_j.view());
            let x_col = super::chol_back_solve(&l, z.view());
            for i in 0..p {
                a_inv[[i, col]] = x_col[i];
            }
        }
        a_inv
    } else {
        let (eigs, eigvecs) = match a_newton.eigh(UPLO::Lower) {
            Ok(p) => p,
            Err(_) => return None,
        };
        let mut a_inv = Array2::<f64>::zeros((p, p));
        for k in 0..p {
            let lam_k = eigs[k];
            if !lam_k.is_finite() || lam_k.abs() < 1e-300 {
                return None;
            }
            let inv_lam_k = 1.0 / lam_k;
            for i in 0..p {
                let vi = eigvecs[[i, k]];
                for j in 0..p {
                    a_inv[[i, j]] += inv_lam_k * vi * eigvecs[[j, k]];
                }
            }
        }
        a_inv
    };

    // a1[i]: v0.x `src/reml/mod.rs:2392-2415`. Newton branch uses
    // w_newton[i] (signed).
    let mut a1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mu_i = mu[i];
        let var_i = family.variance.variance(mu_i).max(1e-300);
        let g_prime_mu = family.link.d_link_dmu(mu_i);
        if g_prime_mu.abs() < 1e-12 {
            continue;
        }
        let v_prime = family.variance.d_variance(mu_i);
        let v1n = v_prime / var_i;
        let v_double_prime = family.variance.d2_variance(mu_i);
        let v2n = v_double_prime / var_i;
        let g_double_prime = family.link.d2_link_dmu(mu_i);
        let g2n = g_double_prime / g_prime_mu;
        let g_triple_prime = family.link.d3_link_dmu(mu_i);
        let g3n = g_triple_prime / g_prime_mu;
        let c_resid = y[i] - mu_i;
        // d(wf·α)/dη for the SAME signed α the score's log|H| uses. Clamping α
        // to 1 where it is ≤ 0 differentiated a different matrix from the one
        // factorised: on an inverse-Gaussian fit 88 of 400 rows have α < 0.
        let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
        let alpha = 1.0 + c_resid * (v1n + g2n);
        let xx = v2n - v1n * v1n + g3n - g2n * g2n;
        let dalpha_dmu = -(v1n + g2n) + c_resid * xx;
        a1[i] = wf * (alpha * (-v1n - 2.0 * g2n) + dalpha_dmu) * g_prime_mu.recip();
    }

    // lev_uw[i] = x_iᵀ A_newton⁻¹ x_i. BLAS form: XAi = X · A⁻¹ (n,p),
    // then lev_uw[i] = Σ_j X[i,j] · XAi[i,j]. Port of mgcv_rust
    // `src/reml/mod.rs:2404-2412` (`let xa = x.dot(&a_inv); ...`).
    let xa: Array2<f64> = x_design.dot(&a_inv);
    let mut lev_uw = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..p {
            s += xa[[i, j]] * x_design[[i, j]];
        }
        lev_uw[i] = s;
    }

    // Per-term b1_k = -λ_k · A_newton⁻¹ · S_k · β; eta1_k = X · b1_k;
    // tr(A_newton⁻¹ · S_k). Multi-smooth port of mgcv_rust
    // `src/reml/mod.rs:2401-2484`.
    let m = s_list.len();
    debug_assert_eq!(
        rho.len(),
        m,
        "lazy_tk_kkt_inputs: rho.len()={} must match s_list.len()={m}",
        rho.len()
    );
    let mut eta1_per_term: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut tr_a_newton_inv_s_per_term: Vec<f64> = Vec::with_capacity(m);
    for k in 0..m {
        let lambda_k = rho[k].exp();
        let s_k = &s_list[k];
        // b1_k = -λ_k · A⁻¹ · S_k · β, then eta1_k = X · b1_k.
        let s_k_beta = s_k.dot(beta);
        let a_inv_s_k_beta = a_inv.dot(&s_k_beta);
        let b1_k = a_inv_s_k_beta.mapv(|v| -lambda_k * v);
        eta1_per_term.push(x_design.dot(&b1_k));
        // tr(A⁻¹ · S_k) = Σ_{i,j} A⁻¹[i,j] · S_k[j,i].
        let mut tr_k = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                tr_k += a_inv[[i, j]] * s_k[[j, i]];
            }
        }
        tr_a_newton_inv_s_per_term.push(tr_k);
    }
    let _ = prior_w; // kept for signature uniformity with the score-weights helper
    let sign_w = Array1::<f64>::ones(n);
    Some(super::TkKKTInputs {
        a1,
        lev_uw,
        eta1_per_term,
        tr_a_newton_inv_s_per_term,
        a_newton_inv: a_inv,
        working_weights_sign: sign_w,
    })
}

/// `crate::traits::InnerSolver` impl for any `Family<L, K, V>` via PIRLS.
///
/// Standard penalised iteratively-reweighted least squares loop:
///
/// ```text
///   loop {
///     z   = η + (y - μ) · g'(μ)           // working response
///     W   = 1 / (V(μ) · g'(μ)²)            // working weights (Fisher info)
///     β   = (X'WX + λS)⁻¹ X'Wz             // backend solve
///     η   = X β
///     μ   = g⁻¹(η)
///     if Δdeviance < tol → done
///   }
/// ```
///
/// Step-halving on β when the deviance increases — same shape as mgcv's
/// `gam.fit3.r:840-890` halving loop. Phase 1 ships the canonical-link
/// Fisher path; non-canonical Newton (full `d²L/dμ²` curvature) is
/// deferred.
///
/// `S: LinearSolver` (default `CholeskySolver`) picks the factorisation
/// backend at the type level — `PirlsInner<L, K, V, LuSolver>` swaps
/// Cholesky for LAPACK LU with no other code changes.
pub struct PirlsInner<
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver = CholeskySolver,
> {
    pub x_design: Array2<f64>,
    pub y: Array1<f64>,
    pub prior_weights: Option<Array1<f64>>,
    /// Per-term penalty blocks `Vec<S_j>` of `(p, p)`. The PIRLS loop
    /// assembles `S_total(ρ) = Σ_j exp(ρ_j) · S_j` per call to `fit(ρ)`.
    /// Single-smooth callers pass `vec![S]`; multi-smooth `Additive`
    /// passes one block per term.
    pub s_list: Vec<Array2<f64>>,
    pub family: Family<L, K, V>,
    pub opts: PirlsOpts,
    pub _solver: PhantomData<S>,
}

#[derive(Clone)]
pub struct PirlsOpts {
    pub max_iters: usize,
    pub dev_rel_tol: f64,
    pub halving_steps: usize,
    /// Initial η = μ_init mapped through the link. `None` → family-specific
    /// default (`(y + 0.5) / 2` for Bernoulli; `y` clamped for Poisson).
    pub eta_init: Option<Array1<f64>>,
    /// Build [`GaussianInnerFit::fisher`] — the extra Fisher-weighted
    /// factorisation that edf and vcov are computed from.
    ///
    /// Off by default because it is consumed ONCE, from the final fit, while
    /// PIRLS runs per score evaluation: building it every time cost ~4ms of a
    /// 56ms scat fit for nothing. The fit drivers turn it on for the final
    /// `inner.fit(&rho_hat)` only.
    pub want_fisher: bool,
}

impl Default for PirlsOpts {
    fn default() -> Self {
        Self {
            max_iters: 50,
            dev_rel_tol: 1e-9,
            halving_steps: 10,
            eta_init: None,
            want_fisher: false,
        }
    }
}

impl<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone, S: LinearSolver> InnerSolver
    for PirlsInner<L, K, V, S>
{
    type Fit = GaussianInnerFit<S>;

    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit> {
        debug_assert_eq!(
            rho.len(),
            self.s_list.len(),
            "PirlsInner: rho length {} must equal s_list length {}",
            rho.len(),
            self.s_list.len()
        );
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        self.pirls_loop(s_total, rho, None)
    }

    /// Warm-start variant: uses `beta_warm` as the initial β if provided
    /// (via `eta = X · beta_warm`), otherwise behaves like `fit`.
    /// Used by `EnvelopeScore::compute_value_no_refresh` to skip cold
    /// initial μ when a NoRefresh IFT propagation is available.
    fn fit_warm(&self, rho: &Array1<f64>, beta_warm: Option<&Array1<f64>>) -> Result<Self::Fit> {
        debug_assert_eq!(rho.len(), self.s_list.len());
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        self.pirls_loop(s_total, rho, beta_warm)
    }

    /// **Single IRLS step** — mgcv R `bam(method="fREML")` single-step
    /// inner port. Build (w, z) at `η = X·β_warm` (or per-family init if
    /// `beta_warm = None`), solve `(X'WX + λS) β = X'Wz` once, populate
    /// a minimal `GaussianInnerFit` and return. No PIRLS loop, no
    /// step-halving, no convergence guard.
    ///
    /// The returned fit's `beta` / `a_factor` are used by Fellner-Schall
    /// to compute `tr(A⁻¹ S_i)` and `β'S_iβ` for the multiplicative λ
    /// update. Score-path-only fields (`mu`, `eta`, `working_weights`,
    /// `dw_deta`, `tk_kkt_inputs`) are populated only as far as is
    /// cheap, since FS doesn't consume them.
    fn fit_single_irls(
        &self,
        rho: &Array1<f64>,
        beta_warm: Option<&Array1<f64>>,
    ) -> Result<Self::Fit> {
        debug_assert_eq!(rho.len(), self.s_list.len());
        let n = self.x_design.nrows();
        let p = self.x_design.ncols();
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        let prior_w: Array1<f64> = match &self.prior_weights {
            Some(w) => w.clone(),
            None => Array1::ones(n),
        };

        // Initial η: warm-start if provided, else family-specific null-init.
        let eta: Array1<f64> = if let Some(b0) = beta_warm {
            debug_assert_eq!(b0.len(), p);
            self.x_design.dot(b0)
        } else {
            let mu_init: Array1<f64> = self.family.loss.initial_mu(self.y.view());
            mu_init.iter().map(|&m| self.family.link.link(m)).collect()
        };
        let mu: Array1<f64> = eta
            .iter()
            .map(|&e| self.family.link.inverse_link(e))
            .collect();

        // Fisher-only IRLS pair (canonical-link auto-promotion — mgcv's
        // fREML uses Fisher uniformly, see audit `pirls/mod.rs:3876-3882`).
        // For canonical-link Poisson + log this is identical to Newton anyway
        // since α ≡ 1 (V' = V, g'' = 0).
        let mut working_weights = Array1::<f64>::zeros(n);
        let mut working_response = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu_i = mu[i];
            let var_i = self.family.variance.variance(mu_i).max(1e-300);
            let g_prime_mu = self.family.link.d_link_dmu(mu_i);
            let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
            working_weights[i] = prior_w[i] * wf;
            working_response[i] = eta[i] + (self.y[i] - mu_i) * g_prime_mu;
        }

        // One WLS solve.
        let xtw = crate::inner::weighted_xt(&self.x_design, &working_weights);
        let xtwx = xtw.dot(&self.x_design);
        let xtwz = xtw.dot(&working_response);
        let mut a = xtwx.clone();
        add_penalty(&mut a, &s_total, 1.0);
        let (factor, beta) = factor_and_solve::<S>(&a, xtwz.view()).map_err(|e| match e {
            GamrsError::SingularSystem(msg) => {
                GamrsError::SingularSystem(format!("PIRLS single-step factor: {msg}"))
            }
            other => other,
        })?;

        // Recompute (η, μ, deviance) at the new β so FS / downstream
        // consumers see a consistent state. O(n·p + n) — cheap.
        let eta_new: Array1<f64> = self.x_design.dot(&beta);
        let mu_new: Array1<f64> = eta_new
            .iter()
            .map(|&e| self.family.link.inverse_link(e))
            .collect();
        let dev_new = self.compute_deviance(&mu_new, &prior_w);

        Ok(GaussianInnerFit {
            beta,
            eta: eta_new,
            mu: mu_new,
            // Factor and its weights travel together — see `a_weights`.
            a_weights: working_weights.clone(),
            // No separate expected curvature on this path — edf/vcov keep
            // using `a_factor`, as before. See `GaussianInnerFit::fisher`.
            fisher: None,
            working_weights,
            working_response,
            deviance: dev_new,
            rss: dev_new,
            n,
            p,
            iterations: 1,
            converged: true, // FS doesn't gate on inner-PIRLS convergence
            a_factor: factor,
            log_det_h_override: None,
            tk_kkt_inputs: None,
            dw_deta: None,
            x_design: None,
        })
    }

    /// Newton-A log|H| at converged β. Computed lazily here so PIRLS
    /// itself doesn't pay the O(p³) cost per inner fit (mgcv_rust pattern
    /// — `fit_pirls_cached` returns no Newton pieces, the score evaluator
    /// at `src/reml/mod.rs:460-483` builds them when needed).
    fn lazy_newton_log_det_h(&self, fit: &Self::Fit, rho: &Array1<f64>) -> Option<f64> {
        if !self.family.loss.use_newton_irls() {
            return None;
        }
        let n = self.x_design.nrows();
        let prior_w: Array1<f64> = self
            .prior_weights
            .clone()
            .unwrap_or_else(|| Array1::ones(n));
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        lazy_newton_log_det_h(
            &self.family,
            &self.y,
            &fit.mu,
            &prior_w,
            &self.x_design,
            &s_total,
        )
    }

    /// Tk·KK' / IFT inputs at converged β. Lazy — see [`InnerSolver::
    /// lazy_tk_kkt_inputs`] docstring.
    fn fisher_log_det_beta_chain(&self, fit: &Self::Fit, rho: &Array1<f64>) -> Option<Vec<f64>> {
        self.fisher_beta_chain(fit, rho)
    }

    fn lazy_tk_kkt_inputs(&self, fit: &Self::Fit, rho: &Array1<f64>) -> Option<super::TkKKTInputs> {
        if !self.family.loss.use_newton_irls() {
            return None;
        }
        let n = self.x_design.nrows();
        let prior_w: Array1<f64> = self
            .prior_weights
            .clone()
            .unwrap_or_else(|| Array1::ones(n));
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        lazy_tk_kkt_inputs(
            &self.family,
            &self.y,
            &fit.mu,
            &fit.beta,
            &prior_w,
            &self.x_design,
            &self.s_list,
            &s_total,
            rho,
        )
    }
}

impl<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone, S: LinearSolver>
    PirlsInner<L, K, V, S>
{
    fn pirls_loop(
        &self,
        s_total: Array2<f64>,
        rho: &Array1<f64>,
        beta_warm: Option<&Array1<f64>>,
    ) -> Result<GaussianInnerFit<S>> {
        // `lambda_eff = 1` since `s_total` already absorbs the per-term λ_j;
        // every `λ · S` site below now reads as `1 · s_total`. Kept named
        // for readability of the mgcv-equivalent algebra.
        let lambda = 1.0_f64;
        let _ = rho; // rho is captured here via s_total; explicit token kept for grep clarity
        let n = self.x_design.nrows();
        let p = self.x_design.ncols();
        let prior_w: Array1<f64> = match &self.prior_weights {
            Some(w) => w.clone(),
            None => Array1::ones(n),
        };

        // Initial η: warm-start with `eta = X · beta_warm` if supplied
        // (Wood 2011 Phase 5 / mgcv_rust `fit_pirls_cached:1077-1094`),
        // otherwise the per-family null-init via `Loss::initial_mu`.
        // `opts.eta_init` still overrides everything for caller-controlled
        // starts (e.g. quantile warm-restart).
        let mut eta: Array1<f64> = if let Some(b0) = beta_warm {
            debug_assert_eq!(b0.len(), p);
            self.x_design.dot(b0)
        } else {
            let mu_init: Array1<f64> = self.family.loss.initial_mu(self.y.view());
            mu_init.iter().map(|&m| self.family.link.link(m)).collect()
        };
        if let Some(e0) = &self.opts.eta_init {
            eta.assign(e0);
        }
        let mut mu: Array1<f64> = eta
            .iter()
            .map(|&e| self.family.link.inverse_link(e))
            .collect();
        let mut dev = self.compute_deviance(&mu, &prior_w);

        let mut beta = Array1::<f64>::zeros(p);
        let mut a_factor_opt: Option<S::Factorization> = None;
        let mut working_weights = Array1::<f64>::ones(n);
        let mut working_response = self.y.clone();
        let mut converged = false;
        let mut iters_used = 0;
        // Penalised deviance at the current accepted state. Starts at the
        // initial-μ deviance (β=0, β'Sβ=0). Tracked alongside `dev` so the
        // mgcv-exact halving (gam.fit3.r:425) can compare pdev-divergence
        // against the previously-accepted state.
        let mut pdev = dev + lambda * beta_sbeta(&s_total, &beta);

        // Newton observed-info IRLS opt-in. Mirrors mgcv R `gam.fit4.r:368-369`
        // (which computes `w = dd$Deta2·0.5`, `z = η − D'_η/D''_η` directly
        // from η-coord deviance derivatives) — algebraically identical to
        // mgcv's `gam.fit3.r` Newton path
        //   `w = wf·α`,  `z = η + (y-μ)·g'(μ)/α`,
        //   wf = 1/(V·g'²),  α = 1 + (y-μ)·(V'/V + g''/g')
        // (port of mgcv_rust `pirls/row_step.rs::compute_irls_wz` with
        // `use_fisher=false`), with the per-row Fisher fallback when α ≤ 0
        // (gam.fit4.r:392-399 sets `w[!good]=0` analogously; the wf form
        // keeps the row's information but reverts to Fisher curvature for
        // that row).
        //
        // Why this matters for the REML score: at convergence the inner
        // step's normal equation is `X'·W·(z − X·β) = 0`. Under Newton-IRLS,
        // each row contributes `W·(z-η) = (y-μ)/(V·g'(μ))`, i.e.
        // `X' W (z - η) = -½ · X' D'_η = -½ · dD/dβ`, so β satisfies
        // `dD(β)/dβ = 0` — the **deviance** is β-stationary. The Fisher
        // fallback rows (α ≤ 0) preserve this identity:
        //   w·(z-η) = wf · (y-μ)·g' = (y-μ)/(V·g') as well. The score-side
        // REML formula uses Newton `log|H|` and Newton-A⁻¹ in the IFT
        // formula; the envelope theorem therefore requires the deviance to
        // be β-stationary, which Newton-IRLS delivers. Fisher-IRLS
        // converges to the same fixed point in the limit but with a
        // different per-iter residual, and on the `log θ` axis of the
        // NegBin shape gradient the leftover gap shows up as 6-23% rel-err
        // vs FD-of-score (boundary test
        // `negbin_multismooth_analytic_grad_matches_fd`). Switching the
        // inner working-weight to Newton closes that gap because the
        // inner-vs-score curvature match becomes exact.
        let use_newton = self.family.loss.use_newton_irls();
        for it in 0..self.opts.max_iters {
            // PIRLS step: build (z, W) per row.
            //   **Observed pair**: families overriding `irls_observed_pair`
            //     get full control of (W, z) per row. TDist uses this to
            //     route through mgcv R's `gam.fit4.r:368-399` observed-W
            //     formula so `fit.a_factor` carries A_obs — what the
            //     Level-2 analytic Hessian assumes.
            //   Fisher: w = prior/(V·g'²),          z = η + (y-μ)·g'(μ)
            //   Newton: w = wf·α·prior (PSD: α>0), z = η + (y-μ)·g'(μ)/α
            //           Fisher fallback when α ≤ 0:
            //             w = wf·prior,            z = η + (y-μ)·g'(μ)
            //   where wf = 1/(V·g'²), α = 1 + (y-μ)·(V'/V + g''/g').
            if let Some((w_obs, z_obs)) = self.family.loss.irls_observed_pair(
                self.y.view(),
                mu.view(),
                eta.view(),
                prior_w.view(),
            ) {
                // Use assign (a memcpy, SIMD-vectorised) instead of indexed
                // for-loop (which bounds-checks every element and may not
                // autovectorise).
                working_weights.assign(&w_obs);
                working_response.assign(&z_obs);
            } else {
                for i in 0..n {
                    let mu_i = mu[i];
                    let var_i = self.family.variance.variance(mu_i).max(1e-300);
                    let g_prime_mu = self.family.link.d_link_dmu(mu_i);
                    let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
                    if !use_newton {
                        working_weights[i] = prior_w[i] * wf;
                        working_response[i] = eta[i] + (self.y[i] - mu_i) * g_prime_mu;
                        continue;
                    }
                    let v_prime = self.family.variance.d_variance(mu_i);
                    let v1n = v_prime / var_i;
                    let g_double_prime = self.family.link.d2_link_dmu(mu_i);
                    let g2n = g_double_prime / g_prime_mu;
                    let c_resid = self.y[i] - mu_i;
                    let alpha = 1.0 + c_resid * (v1n + g2n);
                    if alpha > 0.0 && alpha.is_finite() {
                        working_weights[i] = prior_w[i] * wf * alpha;
                        working_response[i] = eta[i] + c_resid * g_prime_mu / alpha;
                    } else {
                        // Per-row Fisher fallback (mgcv R `gam.fit4.r:392-399`).
                        working_weights[i] = prior_w[i] * wf;
                        working_response[i] = eta[i] + c_resid * g_prime_mu;
                    }
                }
            }

            let (beta_trial, factor_trial) = {
                let xtw = weighted_xt(&self.x_design, &working_weights);
                let xtwx = xtw.dot(&self.x_design);
                let xtwz = xtw.dot(&working_response);
                let mut a = xtwx;
                add_penalty(&mut a, &s_total, lambda);
                let (factor, b) = factor_and_solve::<S>(&a, xtwz.view()).map_err(|e| match e {
                    GamrsError::SingularSystem(msg) => {
                        GamrsError::SingularSystem(format!("PIRLS factor: {msg}"))
                    }
                    other => other,
                })?;
                (b, factor)
            };

            // mgcv-exact three-guard step-halving (gam.fit3.r:382-441).
            // See `halve_until_valid` for the guard sequence; the validity
            // predicate (`eta_mu_valid`) is generic via the Loss trait so
            // the same halving serves every family in the PIRLS path.
            let pdev_old = pdev;
            let iter_one = it == 0;
            let beta_try0 = beta_trial.clone();
            let eta_try0 = self.x_design.dot(&beta_try0);
            let mu_try0: Array1<f64> = eta_try0
                .iter()
                .map(|&e| self.family.link.inverse_link(e))
                .collect();
            let dev_try0 = self.compute_deviance(&mu_try0, &prior_w);
            let pdev_try0 = dev_try0 + lambda * beta_sbeta(&s_total, &beta_try0);

            let recompute = |b: &Array1<f64>| {
                let e = self.x_design.dot(b);
                let m: Array1<f64> = e
                    .iter()
                    .map(|&ev| self.family.link.inverse_link(ev))
                    .collect();
                let d = self.compute_deviance(&m, &prior_w);
                let pd = d + lambda * beta_sbeta(&s_total, b);
                (e, d, pd, Some(m))
            };
            let is_invalid = |e: &Array1<f64>, m: Option<&Array1<f64>>| -> bool {
                let m = m.expect("PIRLS halving always provides μ");
                !self.eta_mu_valid(e, m)
            };

            let (beta_try, eta_try, dev_try, pdev_try, mu_try_opt, accepted) = halve_until_valid(
                beta_try0,
                &beta,
                eta_try0,
                dev_try0,
                pdev_try0,
                Some(mu_try0),
                pdev_old,
                iter_one,
                recompute,
                is_invalid,
            );

            if accepted {
                let dev_change = (dev - dev_try).abs() / (dev.abs() + 1e-30);
                beta = beta_try;
                eta = eta_try;
                mu = mu_try_opt.expect("PIRLS halving always returns μ");
                a_factor_opt = Some(factor_trial);
                if it > 0 && dev_change < self.opts.dev_rel_tol {
                    converged = true;
                }
                dev = dev_try;
                pdev = pdev_try;
            }
            iters_used = it + 1;
            if !accepted {
                // 100 halvings exhausted and still invalid — bail with the
                // last successful state. (Same behaviour as v0.x's revert.)
                break;
            }
            if converged {
                break;
            }
        }

        // If the loop never accepted a step, factor whatever we have at the
        // current (zero) β so the score still receives a usable factor.
        let a_factor = match a_factor_opt {
            Some(f) => f,
            None => {
                // Rebuild A at the current β and factor it — initial β=0
                // makes this `X' diag(prior) X + λS` for unweighted PIRLS.
                let xtw = weighted_xt(&self.x_design, &prior_w);
                let xtwx = xtw.dot(&self.x_design);
                let mut a = xtwx;
                add_penalty(&mut a, &s_total, lambda);
                let max_diag = a.diag().iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
                for i in 0..p {
                    a[[i, i]] += 1e-12 * max_diag;
                }
                S::factorize(a)?
            }
        };

        // Final pass: refresh `(working_weights, working_response)` AT the
        // converged μ so downstream consumers (working_rss, dw_deta,
        // score-side log|H| / tk_kkt) see the same `(w, z)` the next outer
        // probe would assemble. Without this they lag by one IRLS iter (μ
        // is updated at the end of each iter, while (w, z) were built at
        // the top from the previous μ). Mirrors `OcatInner::ocat_loop`'s
        // final pass at `src/inner/gam_fit5.rs:220-225`. Harmless for
        // Fisher (β converged ⇒ μ unchanged ⇒ same (w, z)), load-bearing
        // for Newton and observed-pair paths (the A⁻¹ materialised in
        // `compute_tk_kkt_inputs` and `fit.a_factor` is built from the SAME μ).
        if let Some((w_obs, z_obs)) = self.family.loss.irls_observed_pair(
            self.y.view(),
            mu.view(),
            eta.view(),
            prior_w.view(),
        ) {
            working_weights.assign(&w_obs);
            working_response.assign(&z_obs);
        } else {
            for i in 0..n {
                let mu_i = mu[i];
                let var_i = self.family.variance.variance(mu_i).max(1e-300);
                let g_prime_mu = self.family.link.d_link_dmu(mu_i);
                let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
                if !use_newton {
                    working_weights[i] = prior_w[i] * wf;
                    working_response[i] = eta[i] + (self.y[i] - mu_i) * g_prime_mu;
                    continue;
                }
                let v_prime = self.family.variance.d_variance(mu_i);
                let v1n = v_prime / var_i;
                let g_double_prime = self.family.link.d2_link_dmu(mu_i);
                let g2n = g_double_prime / g_prime_mu;
                let c_resid = self.y[i] - mu_i;
                let alpha = 1.0 + c_resid * (v1n + g2n);
                if alpha > 0.0 && alpha.is_finite() {
                    working_weights[i] = prior_w[i] * wf * alpha;
                    working_response[i] = eta[i] + c_resid * g_prime_mu / alpha;
                } else {
                    working_weights[i] = prior_w[i] * wf;
                    working_response[i] = eta[i] + c_resid * g_prime_mu;
                }
            }
        }

        // Make `a_factor` the matrix the SCORE should differentiate.
        //
        // Everything downstream — `log_det_a`, `trace_a_inv`, `a_inv`, and so
        // the gradient's `h_diag` and IFT bracket — reads this one factor, so
        // rebuilding it here is the single seam that keeps the criterion and
        // all of its derivatives consistent. The loop's factor was built from
        // the working weight, which for an observed→expected switch family is
        // NOT the Hessian of the penalised deviance on the outlier rows (and
        // is one μ-step stale besides — the final pass above refreshed
        // `working_weights` but not the factor).
        //
        // Cholesky can legitimately fail here: negative rows are the point,
        // and mgcv's own rule is that individual `w_i < 0` are fine only while
        // `X'WX + E'E` stays PD (`gdi.c`'s `pls_fit1` returns `n < 0`
        // otherwise and `gam.fit4.r` retries with Fisher). On failure we keep
        // the loop's factor, which is that same fallback.
        // Keep the factor and the weights it was built from together — see
        // `GaussianInnerFit::a_weights`.
        let (a_factor, a_weights) =
            match family_observed_score_weights(&self.family, &self.y, &eta, &prior_w) {
                Some(w_score) => {
                    let xtw = weighted_xt(&self.x_design, &w_score);
                    let mut a = xtw.dot(&self.x_design);
                    add_penalty(&mut a, &s_total, lambda);
                    // This factorisation IS the PD decision now: mgcv permits
                    // negative `w_i` but needs `X'WX + E'E` positive definite
                    // overall (`gdi.c`'s `pls_fit1` returns n < 0 otherwise and
                    // `gam.fit4.r` retries with Fisher). On failure we keep the
                    // working-weight factor, which is that same fallback — and
                    // it means the objective changed mid-optimisation, so it is
                    // counted rather than silently taken.
                    match S::factorize(a) {
                        Ok(f) => {
                            OBS_PD_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            (f, w_score)
                        }
                        Err(_) => {
                            OBS_PD_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            (a_factor, working_weights.clone())
                        }
                    }
                }
                None => (a_factor, working_weights.clone()),
            };

        // The Fisher pair for edf / vcov — see `GaussianInnerFit::fisher`.
        // Built here because this is where the family and the penalty are both
        // in hand; costs one extra p×p assembly + factorisation per inner fit,
        // and p is the basis dimension.
        let fisher = if !self.opts.want_fisher {
            None
        } else {
            self.family
                .loss
                .expected_curvature_weights(self.y.view(), eta.view(), Some(prior_w.view()))
                .filter(|w| w.iter().all(|v| v.is_finite()))
                .and_then(|w_f| {
                    let xtw = weighted_xt(&self.x_design, &w_f);
                    let mut a = xtw.dot(&self.x_design);
                    add_penalty(&mut a, &s_total, lambda);
                    S::factorize(a).ok().map(|f| (f, w_f))
                })
        };

        // For PIRLS at convergence: rss-like quantity for downstream code is
        // the working-RSS `Σ W·(z - X·β)²`, which mgcv calls `dev_num` (it
        // matches the GLM deviance at convergence for canonical links).
        let mut working_rss = 0.0;
        for i in 0..n {
            let r = working_response[i] - eta[i];
            working_rss += working_weights[i] * r * r;
        }

        // **Newton log|H| and Tk·KK' are computed LAZILY by consumers** —
        // not here. Port of mgcv_rust `src/pirls/mod.rs::fit_pirls_cached`
        // (lines 1020-1240): mgcv_rust's PIRLS returns only `(β, μ,
        // working_weights)` and the REML score evaluator builds Newton-A
        // pieces at gradient time (mgcv_rust `src/reml/mod.rs:460-483`
        // for log|H|; `src/reml/mod.rs:2347-2487` for Tk·KK'). Doing it
        // here was an O(p³) eigh per inner fit — 25-30× perf regression
        // on the NegBin bench. See `super::pirls::lazy_newton_log_det_h`
        // and `super::pirls::lazy_tk_kkt_inputs` for the moved code.
        let log_det_h_override: Option<f64> = None;
        let tk_kkt_inputs: Option<super::TkKKTInputs> = None;
        // Keep `rho` referenced (was previously consumed by
        // `compute_tk_kkt_inputs(... rho)` — kept for grep-symmetry).
        let _ = rho;

        // Per-obs `∂W/∂η` for the analytic outer-Newton Hessian's W-chain
        // term. **Always the Fisher W derivative** — `W_F(μ) = prior_w /
        // (V(μ)·g'(μ)²)` — even when `use_newton_irls = true`. The
        // analytic Hessian consumer is `EnvelopeScore` which is only
        // wired for canonical-link families (Bernoulli, Poisson, Gamma+inv,
        // Gaussian — all Fisher == Newton) and for InverseGaussian + log
        // (Newton-IRLS, but the test floor stays below the Newton vs
        // Fisher H gap so the simpler Fisher derivative is correct
        // enough). The shape-aware path (NegBin, scat, Tweedie) does not
        // read `dw_deta`. Keeping the FD on the Fisher form makes the
        // `VarianceFn::d_variance` default-zero impls work for canonical
        // families without subtle mis-scaling.
        //
        // We differentiate THAT function — not a hand-expanded `V'/V +
        // 2g''/g'` form — by a tight central difference in μ, then chain
        // through `dμ/dη = 1/g'(μ)`. The FD here only calls `variance` /
        // `d_link_dmu`, both always correct.
        let mut dw_deta = Array1::<f64>::zeros(n);
        let w_of_mu = |m: f64, pw: f64| -> f64 {
            let v = self.family.variance.variance(m).max(1e-300);
            let gp = self.family.link.d_link_dmu(m);
            pw / (v * gp * gp)
        };
        for i in 0..n {
            let mu_i = mu[i];
            let pw = prior_w[i];
            let g_prime = self.family.link.d_link_dmu(mu_i);
            // Scale the μ-step to the magnitude of μ for conditioning.
            let hmu = 1e-6 * mu_i.abs().max(1e-3);
            let dw_dmu = (w_of_mu(mu_i + hmu, pw) - w_of_mu(mu_i - hmu, pw)) / (2.0 * hmu);
            dw_deta[i] = dw_dmu / g_prime; // dμ/dη = 1/g'(μ)
        }

        Ok(GaussianInnerFit::<S> {
            beta,
            eta,
            mu,
            working_weights,
            working_response,
            deviance: dev,
            rss: working_rss,
            n,
            p,
            iterations: iters_used,
            converged,
            a_factor,
            a_weights,
            fisher,
            log_det_h_override,
            tk_kkt_inputs,
            dw_deta: Some(dw_deta),
            x_design: Some(self.x_design.clone()),
        })
    }

    fn compute_deviance(&self, mu: &Array1<f64>, prior_w: &Array1<f64>) -> f64 {
        let mut s = 0.0;
        for i in 0..self.y.len() {
            s += prior_w[i] * self.family.loss.deviance_per_obs(self.y[i], mu[i]);
        }
        s
    }

    /// Generic (η, μ)-validity check — gamrs's link-/family-agnostic analogue
    /// of mgcv's `family$valideta` and `family$validmu`. Defined in terms of
    /// the existing trait surface (no new "what family is this" dispatch):
    ///   - η: every entry finite (catches `link(μ)`-divergence; mgcv's
    ///     `binomial()$valideta` for instance accepts any finite η).
    ///   - μ: every entry finite (catches `inverse_link` blowing up).
    ///   - deviance per obs is finite for every (y_i, μ_i) — this is the
    ///     family's own validity statement: Bernoulli's μ ∈ (0, 1) and
    ///     Poisson/Gamma/IG's μ > 0 each emit non-finite deviance outside
    ///     their support (because `log(0)` / division by zero / negative
    ///     log argument). Using the Loss as the validity oracle keeps the
    ///     halving generic over all `Loss + Link + VarianceFn` triples.
    fn eta_mu_valid(&self, eta: &Array1<f64>, mu: &Array1<f64>) -> bool {
        for i in 0..eta.len() {
            if !eta[i].is_finite() || !mu[i].is_finite() {
                return false;
            }
            if !self
                .family
                .loss
                .deviance_per_obs(self.y[i], mu[i])
                .is_finite()
            {
                return false;
            }
        }
        true
    }
}

impl<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone, S: LinearSolver> PirlsInner<L, K, V, S> {
    /// See `InnerSolver::fisher_log_det_beta_chain`.
    pub(crate) fn fisher_beta_chain(&self, fit: &GaussianInnerFit<S>, rho: &Array1<f64>) -> Option<Vec<f64>> {
        use ndarray_linalg::Solve;
        if self.family.loss.use_newton_irls() {
            return None;
        }
        let n = self.x_design.nrows();
        let prior: Array1<f64> = self.prior_weights.clone().unwrap_or_else(|| Array1::ones(n));
        let mut dw_deta = Array1::<f64>::zeros(n);
        let mut w_obs = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu = fit.mu[i];
            let v = self.family.variance.variance(mu).max(1e-300);
            let g1 = self.family.link.d_link_dmu(mu);
            if g1.abs() < 1e-300 {
                continue;
            }
            let v1n = self.family.variance.d_variance(mu) / v;
            let g2n = self.family.link.d2_link_dmu(mu) / g1;
            let wf = prior[i] / (v * g1 * g1);
            dw_deta[i] = wf * (-v1n - 2.0 * g2n) / g1;
            w_obs[i] = wf * (1.0 + (self.y[i] - mu) * (v1n + g2n));
        }
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        let wx = &self.x_design * &w_obs.view().insert_axis(ndarray::Axis(1));
        let a_obs = self.x_design.t().dot(&wx) + &s_total;
        let a_inv = fit.a_inv();
        let xa = self.x_design.dot(&a_inv);
        let lev: Array1<f64> = (0..n).map(|i| xa.row(i).dot(&self.x_design.row(i))).collect();
        let mut out = Vec::with_capacity(self.s_list.len());
        for (k, s_k) in self.s_list.iter().enumerate() {
            let rhs = s_k.dot(&fit.beta).mapv(|v| -rho[k].exp() * v);
            let db = a_obs.solve(&rhs).ok()?;
            let eta1 = self.x_design.dot(&db);
            out.push((&dw_deta * &eta1 * &lev).sum());
        }
        Some(out)
    }
}
