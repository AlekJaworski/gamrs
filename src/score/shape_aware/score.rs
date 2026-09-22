//! `ShapeAwareEnvelopeScore` struct, type aliases, the `ScoreDerivatives`
//! trait impl, plus the orchestrating `fit_inner_at` / `score_value`
//! helpers used by both the gradient and Hessian paths.

use std::cell::RefCell;
use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::{Family, IdentityLink, OcatLoss, OcatVariance};
use crate::inner::{CholeskySolver, GaussianInnerFit, LinearSolver, PirlsOpts};
use crate::traits::{CoordsKind, InnerSolver, Link, Loss, ScoreDerivatives, VarianceFn};

use super::super::profile::{FixedAtOneProfile, OwnedByLossProfile, Profile};
use super::builder::{OcatInnerBuilder, PirlsInnerBuilder, ShapeInnerBuilder};

/// Cached state from the last accepted full eval (`value_grad_hess` or
/// `compute_value_grad_hess_rho_only_with_fit`). Feeds the NoRefresh IFT
/// line-search shortcut: at trial ρ, the score propagates
/// `β_trial = β + Σ_k b1[:,k]·Δρ_k` and runs ONE working-pair IRLS step
/// to get (w, z, X'WX) without converging PIRLS.
///
/// Port of mgcv_rust's `warm_state` RefCell at
/// `gam_optimized.rs:1408-1414` (β + b1 + λ tuple).
#[doc(hidden)]
#[derive(Clone)]
pub struct AcceptedState {
    /// Converged β at the last accepted point.
    pub beta: Array1<f64>,
    /// First-order IFT derivative `b1[:, k] = ∂β/∂ρ_k = -λ_k · A⁻¹ · S_k · β`
    /// at the last accepted point. Shape (p, n_terms).
    pub b1: Array2<f64>,
    /// λ vector at the last accepted point. Length n_terms.
    pub lambda: Vec<f64>,
    /// Shape parameters at the last accepted point. Length
    /// `family.n_shape_params()`. NoRefresh only fires when the trial
    /// shape exactly matches this — IFT b1 doesn't include shape chain.
    pub shape_params: Vec<f64>,
}

/// Frozen-β̂ context shared between `eval_grad_with_fit` and
/// `eval_grad_frozen_beta`. Holds the converged-inner quantities that
/// stay constant across ±h shape probes.
///
/// Per-term vectors (`bsb_per_term`, `tr_hinv_s_per_term`, length T) feed
/// the per-ρ_j envelope gradient at frozen β̂; `bsb_total = Σ_j λ_j · bsb_j`
/// feeds the φ formula / shape gradient (where the family sees the
/// penalty contribution aggregated). `deviance`, `phi_center`,
/// `n_minus_mp` are scalars.
#[derive(Clone)]
pub(super) struct FrozenBetaCtx {
    pub(super) bsb_per_term: Vec<f64>,
    pub(super) tr_hinv_s_per_term: Vec<f64>,
    pub(super) bsb_total: f64,
    pub(super) phi_center: f64,
    pub(super) n_minus_mp: f64,
    pub(super) deviance: f64,
}

/// REML/LAML closed-form envelope score for families with shape parameters.
///
/// θ layout: `[log λ, shape_params...]` where `shape_params` has length
/// `family.n_shape_params()`. The score owns a "base" family that's
/// cloned and updated per probe — propagating to both the Loss and the
/// VarianceFn via `Family::set_shape_params` (architecture-assumptions.md
/// §E2). The `B` type parameter picks the inner solver (PIRLS vs ocat).
///
/// The `P: Profile<L>` type parameter picks the dispersion convention at
/// score-construction time (closes audit §80 — replaces the prior runtime
/// `family.loss.fixed_dispersion().unwrap_or(1.0)` branch):
///
/// - `FixedAtOneProfile` — Ocat, scat/TDist, NegBin (φ ≡ 1).
/// - `OwnedByLossProfile` — Tweedie (φ lives on the Loss, updated per
///   outer probe via `set_shape_params`).
///
/// Gradient strategy: analytic envelope for the `log λ` component (same
/// as Gaussian/Bernoulli), `Loss::analytic_shape_score_gradient` for the
/// shape components when the family supplies it, central FD otherwise.
pub struct ShapeAwareEnvelopeScore<L, K, V, B, P, S = CholeskySolver>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    pub x_design: Array2<f64>,
    pub y: Array1<f64>,
    pub prior_weights: Option<Array1<f64>>,
    /// Per-term penalty blocks. Multi-smooth in 0.2 — joint outer-Newton
    /// optimises `θ = [ρ₁, …, ρ_T, shape_0, …, shape_{n_shape-1}]` where
    /// `T = s_list.len()`. The shape-aware drivers (TDist/scat, NegBin,
    /// Tweedie, Ocat) all flow through this score type unchanged.
    pub s_list: Vec<Array2<f64>>,
    /// "Base" family — cloned per probe, then shape params updated from θ.
    pub family_base: Family<L, K, V>,
    /// Per-term rank, length `T = s_list.len()`.
    pub rank_s_list: Vec<usize>,
    pub mp: usize,
    pub log_pseudo_det_s_list: Vec<f64>,
    pub coords: CoordsKind,
    pub pirls_opts: PirlsOpts,
    pub inner_builder: B,
    pub profile: P,
    pub _solver: PhantomData<S>,
    /// Last-accepted (β, b1, λ, shape) for the NoRefresh IFT line-search
    /// shortcut. `RefCell` because `ScoreDerivatives` uses `&self` —
    /// updates happen inside `value_grad_hess` / `compute_value_grad_hess_*`
    /// at converged β̂, line-search probes read it via
    /// `compute_value_no_refresh`. `None` until the first full eval lands.
    #[doc(hidden)]
    pub accepted_state: RefCell<Option<AcceptedState>>,
    /// Cell-based diagnostic counters. Bumped by outer.rs, profile_shape.rs,
    /// fit_inner_at, and compute_value_no_refresh. Read after fit via
    /// `score.stats().unwrap().snapshot()`.
    pub stats: crate::stats::FitStats,
    /// Last converged `η̂` from a prior PIRLS solve, kept across outer
    /// Newton iters as a warm-start seed for the next `fit_inner_at`.
    /// Cold-starting every call from η=link(initial_μ) costs significant
    /// inner-iter count on nonlinear PIRLS paths (scat / TDist's observed-W
    /// inner loop: 10 iters/call cold vs 2-3 warm). mgcv R's `gam.fit4`
    /// keeps `coefold`/`etaold` across outer iters and reuses them as
    /// the inner-PIRLS init.
    pub last_eta: RefCell<Option<Array1<f64>>>,
}

/// PIRLS-driven shape-aware score — what TDist/scat, NegBin use.
/// (φ fixed at 1 — for Tweedie use `ShapeAwarePirlsScoreOwnedPhi`.)
pub type ShapeAwarePirlsScore<L, K, V> =
    ShapeAwareEnvelopeScore<L, K, V, PirlsInnerBuilder, FixedAtOneProfile, CholeskySolver>;

/// PIRLS-driven shape-aware score with φ read live off the Loss — Tweedie.
pub type ShapeAwarePirlsScoreOwnedPhi<L, K, V> =
    ShapeAwareEnvelopeScore<L, K, V, PirlsInnerBuilder, OwnedByLossProfile, CholeskySolver>;

/// Ocat-driven shape-aware score — what `fit_ocat_cr` uses (φ ≡ 1).
pub type ShapeAwareOcatScore = ShapeAwareEnvelopeScore<
    OcatLoss,
    IdentityLink,
    OcatVariance,
    OcatInnerBuilder,
    FixedAtOneProfile,
    CholeskySolver,
>;

impl<L, K, V, B, P, S> ScoreDerivatives for ShapeAwareEnvelopeScore<L, K, V, B, P, S>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    fn dim(&self) -> usize {
        // 0.2: multi-smooth — θ = [ρ_1, …, ρ_T, shape_0, …, shape_{n_shape-1}].
        self.s_list.len() + self.family_base.n_shape_params()
    }

    fn coords(&self) -> CoordsKind {
        self.coords.clone()
    }

    fn value(&self, theta: &Array1<f64>) -> Result<f64> {
        self.compute_value(theta)
    }

    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        self.compute_value_grad(theta)
    }

    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        // Coupled value/grad/Hess at θ. The Hessian path uses ONE inner
        // PIRLS fit at θ (the same one that produces (v, g)) and then
        // does central FD on the analytic gradient with **frozen β̂** at
        // the ±h probes. This is v0.x's `tweedie_theta_grad_hess_analytic`
        // recipe generalised — at converged β̂ the gradient's β-chain is
        // zero by envelope theorem, so FD-on-grad-at-frozen-β̂ is exact
        // to O(h) in the Hessian (same accuracy as FD-on-grad with
        // re-converged β̂, at 1 PIRLS solve instead of 2d+1).
        self.compute_value_grad_hess_analytical(theta)
    }

    fn axis_step_caps(&self) -> Option<Vec<f64>> {
        // ρ axes: 5.0 per axis (matches gamrs's global default, mgcv's
        // unspecified-default Newton step magnitude in log-λ space).
        // Shape axes: family-supplied (e.g. ocat θ → 0.5, Tweedie p_t → 2.0).
        let n_terms = self.s_list.len();
        let mut caps: Vec<f64> = vec![5.0; n_terms];
        caps.extend(self.family_base.loss.shape_axis_step_caps());
        Some(caps)
    }

    fn axis_bounds(&self) -> Option<Vec<(f64, f64)>> {
        // ρ axes: bounded above where the criterion stops being computable.
        // `A = X'WX + Σλ_jS_j` is factorised as is, so once λ_j·S_j dwarfs X'WX
        // the penalty's null-space directions (a smooth's linear part is not
        // axis-aligned) are recovered from a cancellation of order
        // eps·λ_j·max diag S_j / max diag X'X. Past `grad_tol / eps` that
        // round-off exceeds the gradient tolerance, the REML surface acquires
        // a slope of its own, and the outer Newton follows it: on a n=66 scat
        // draw the gradient at fixed shape read −6.8e-6 at ρ=20, −1.1e-2 at
        // ρ=30, −11 at ρ=38.6 (the true gradient decays like 1/λ), and the
        // fit came back at λ=5.7e16 with edf 2.63 on what is a straight line
        // at edf 2.0000. mgcv avoids the region by reparameterising
        // (Wood 2011 Appendix B); until gamrs does, the box ends where the
        // arithmetic does. The lower bound stays wide: λ→0 costs nothing.
        let p = self.x_design.ncols();
        let xtx_diag_max = (0..p)
            .map(|c| self.x_design.column(c).mapv(|v| v * v).sum())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let cond_limit = crate::outer::NewtonOpts::default().grad_tol / f64::EPSILON;
        let mut bnds: Vec<(f64, f64)> = self
            .s_list
            .iter()
            .map(|s_j| {
                let s_diag_max = s_j.diag().iter().cloned().fold(0.0_f64, f64::max);
                let hi = if s_diag_max > 0.0 {
                    (cond_limit * xtx_diag_max / s_diag_max).ln().min(50.0)
                } else {
                    50.0
                };
                (-50.0, hi)
            })
            .collect();
        bnds.extend(self.family_base.loss.shape_axis_bounds());
        Some(bnds)
    }

    fn stats(&self) -> Option<&crate::stats::FitStats> {
        Some(&self.stats)
    }
}

impl<L, K, V, B, P, S> ShapeAwareEnvelopeScore<L, K, V, B, P, S>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Build a family with the shape params from θ and run the inner solve.
    /// Returns the inner fit plus the rebuilt family (so the score body
    /// can read `loss.saturated_log_lik` / `loss.fixed_dispersion`
    /// consistent with the params).
    pub(super) fn fit_inner_at(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(GaussianInnerFit<S>, Family<L, K, V>)> {
        let n_terms = self.s_list.len();
        let n_shape = self.family_base.n_shape_params();
        debug_assert_eq!(
            theta.len(),
            n_terms + n_shape,
            "θ has wrong length for shape-aware score"
        );
        let rho_slice: Array1<f64> = theta.slice(ndarray::s![..n_terms]).to_owned();
        let shape_slice: Vec<f64> = theta.iter().skip(n_terms).copied().collect();

        let mut family = self.family_base.clone();
        if n_shape > 0 {
            family.set_shape_params(&shape_slice);
        }
        // Warm-start PIRLS from the previous outer iter's converged η̂.
        // Cold-starting from β=0 (PIRLS default) costs significant inner
        // iters on nonlinear paths like TDist's observed-W (≈ 10 iters
        // cold vs 2-3 warm). The eta_init opts field is the supported
        // entry point — pirls.rs:600 unconditionally honours it. Only
        // override when shape dims match; resizing X or s_list mid-fit
        // would invalidate the stashed η.
        let mut opts = self.pirls_opts.clone();
        if opts.eta_init.is_none() {
            if let Some(eta_warm) = self.last_eta.borrow().as_ref() {
                if eta_warm.len() == self.y.len() {
                    opts.eta_init = Some(eta_warm.clone());
                }
            }
        }
        let inner = {
            let _t = crate::profile::scoped("fit_inner_build");
            self.inner_builder.build(
                family.clone(),
                self.x_design.clone(),
                self.y.clone(),
                self.prior_weights.clone(),
                self.s_list.clone(),
                opts,
            )
        };
        let fit = {
            let _t = crate::profile::scoped("fit_inner_pirls");
            inner.fit(&rho_slice)?
        };
        self.stats.record_pirls_call(fit.iterations);
        // Stash the converged η̂ for the next outer iter's warm-start.
        *self.last_eta.borrow_mut() = Some(fit.eta.clone());
        Ok((fit, family))
    }

    /// Assemble the score value from a converged inner fit at the current θ.
    ///
    /// Extended-family REML/LAML (mgcv `gam.fit5.r:~1003`, v0.x
    /// `src/reml/mod.rs:483`):
    ///   REML = D/(2φ) - Σ ls + log|H|/2 - log|λS|+/2 - Mp/2·log(2πφ)
    ///
    /// For ocat: `OcatLoss::fixed_dispersion() = Some(1.0)` and
    /// `OcatLoss::saturated_log_lik = 0`, so the formula collapses to
    /// `D/2 + log|H|/2 - log|λS|+/2 - Mp/2·log(2π)` — the same formula
    /// `ShapeAwareOcatScore` used pre-unification.
    pub(super) fn score_value(
        &self,
        fit: &GaussianInnerFit<S>,
        family: &Family<L, K, V>,
        rho_slice: &[f64],
    ) -> f64 {
        let n_terms = self.s_list.len();
        debug_assert_eq!(rho_slice.len(), n_terms);

        // Per-family mgcv-style rank adjustment (ocat: −1; others: 0).
        let rank_adj = family.loss.score_rank_adjustment();
        // Per-term bsb_j = β'S_j β + aggregate via λ_j. dp = D + Σ_j λ_j β'S_jβ.
        let mut bsb_total = 0.0_f64;
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = self.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let lambda_j = rho_slice[j].exp();
            bsb_total += lambda_j * bsb_j;
            let adj_rank_j = ((self.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            log_det_lambda_s += adj_rank_j * rho_slice[j] + self.log_pseudo_det_s_list[j];
        }
        let dp = fit.deviance + bsb_total;
        // tr(H⁻¹X'WX) for the Profile signature — unused by both
        // FixedAtOneProfile and OwnedByLossProfile in the shape-aware
        // path, so the cheap `p` upper-bound is fine.
        let tr_hinv_xtwx = fit.p as f64;
        // Profile sees aggregated `bsb_total` with `lambda = 1` (matches
        // EnvelopeScore convention — envelope.rs:266-273).
        let phi =
            match self
                .profile
                .dispersion(&family.loss, fit, 1.0, bsb_total, tr_hinv_xtwx, self.mp)
            {
                Some(p) => p,
                None => return 1e12,
            };
        // Use the Newton-W `log|H|` when the family opts into
        // `Loss::use_newton_irls()` (NegBin / InverseGaussian / similar
        // non-canonical-link). Computed **lazily** here — not inside
        // PIRLS — to match mgcv_rust's pattern (its `fit_pirls_cached`
        // returns no Newton-A pieces; the score evaluator at
        // `src/reml/mod.rs:460-483` builds them on demand). Cholesky-
        // first via `lazy_newton_log_det_h` is O(p³/3) on the PSD α>0
        // path (NegBin always; IG most of the time) — vs the O(3p³)
        // eigh that previously ran per inner fit. Falls back to the
        // Fisher A factor's `log|H|` for canonical-link families and
        // when the Newton path bails out.
        // One definition of the `log|H|` precedence — see
        // `pirls::score_log_det_h`. The gate keeps the fast path free: no
        // `s_total` built when neither the observed nor the Newton path applies.
        let log_det_h = if crate::inner::pirls::score_log_det_h_applies(&family.loss) {
            let prior_w = self
                .prior_weights
                .clone()
                .unwrap_or_else(|| ndarray::Array1::ones(fit.n));
            let s_total = crate::design::combined_s(
                &self.s_list,
                &ndarray::Array1::from(rho_slice.to_vec()),
                self.x_design.ncols(),
            );
            crate::inner::pirls::score_log_det_h(
                family,
                &self.y,
                &fit.eta,
                &fit.mu,
                &prior_w,
                &self.x_design,
                &s_total,
                fit,
            )
        } else {
            fit.log_det_a()
        };
        let ls_sum: f64 = self
            .y
            .iter()
            .map(|&yi| family.loss.saturated_log_lik(yi, phi))
            .sum();
        crate::score::reml_score_from_parts(dp, phi, self.mp, log_det_h, log_det_lambda_s, ls_sum)
    }

    /// Evaluate the REML score at `theta` using a **FROZEN inner fit**
    /// from a previous ρ probe (typically the center of the current
    /// outer iter). Deviance and `saturated_log_lik` are recomputed at the
    /// perturbed shape (since they depend on θ via `V(μ; θ)` and the NB
    /// likelihood); `log_det_h` is recomputed on frozen μ via the Newton-W
    /// path (which sees the new family's `V(μ; θ)`); β̂, μ̂, η̂, `bsb`,
    /// `tr(H⁻¹ S_j)` are reused as-is.
    ///
    /// Port of mgcv_rust `src/reml/mod.rs:693-729` `OuterLinearCache::
    /// score_at_theta_with_phi_hint`. Pre-condition: `fit` is the
    /// converged inner fit at `θ_center = [ρ_slice; shape_center]` where
    /// `ρ_slice = θ[..n_terms]` (the ρ-block matches; only the shape
    /// trailing slice may differ). Caller is responsible for keeping that
    /// invariant — typically by computing `fit` via `fit_inner_at(&theta)`
    /// once per outer iter (center probe), then reusing it for the ±h
    /// FD probes and the line-search candidate evaluations on the shape
    /// axis only.
    ///
    /// PIRLS economy: ZERO PIRLS solves. Just a `lazy_newton_log_det_h`
    /// O(p³) factorisation + an O(n) deviance / ls recomputation. Matches
    /// mgcv_rust's per-FD-probe cost (which is the dominant remaining
    /// per-iter cost vs the full-PIRLS path for NegBin).
    pub(crate) fn score_value_frozen_beta(
        &self,
        fit: &GaussianInnerFit<S>,
        theta: &Array1<f64>,
    ) -> f64 {
        let _t = crate::profile::scoped("frozen_beta_probe");
        let n_terms = self.s_list.len();
        let n_shape = self.family_base.n_shape_params();
        debug_assert_eq!(theta.len(), n_terms + n_shape);
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let shape_slice: Vec<f64> = theta.iter().skip(n_terms).copied().collect();

        let mut family = self.family_base.clone();
        if n_shape > 0 {
            family.set_shape_params(&shape_slice);
        }

        let rank_adj = family.loss.score_rank_adjustment();

        // bsb_total (β'S_jβ from frozen β̂) and log|λS|+ pieces.
        let mut bsb_total = 0.0_f64;
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = self.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let lambda_j = rho_slice[j].exp();
            bsb_total += lambda_j * bsb_j;
            let adj_rank_j = ((self.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            log_det_lambda_s += adj_rank_j * rho_slice[j] + self.log_pseudo_det_s_list[j];
        }

        // Re-compute deviance from frozen μ̂ + new family θ.
        // (mgcv_rust:703 `let dev_numerator = glm_deviance(self.y_for_ls, &mu, family)`.)
        let deviance_new: f64 = self
            .y
            .iter()
            .zip(fit.mu.iter())
            .map(|(&yi, &mui)| family.loss.deviance_per_obs(yi, mui))
            .sum();

        let dp = deviance_new + bsb_total;
        let tr_hinv_xtwx = fit.p as f64;
        let phi =
            match self
                .profile
                .dispersion(&family.loss, fit, 1.0, bsb_total, tr_hinv_xtwx, self.mp)
            {
                Some(p) => p,
                None => return 1e12,
            };

        // log|H| — recompute via Newton-W on FROZEN μ̂ with the NEW family
        // (V(μ; θ) shifts → W shifts). For canonical-link families this
        // returns the same value as the Fisher-W path; for NegBin (Newton
        // IRLS = true) the recompute is essential.
        // One definition of the `log|H|` precedence — see
        // `pirls::score_log_det_h`. The gate keeps the fast path free: no
        // `s_total` built when neither the observed nor the Newton path applies.
        let log_det_h = if crate::inner::pirls::score_log_det_h_applies(&family.loss) {
            let prior_w = self
                .prior_weights
                .clone()
                .unwrap_or_else(|| Array1::ones(fit.n));
            let s_total = crate::design::combined_s(
                &self.s_list,
                &Array1::from(rho_slice.clone()),
                self.x_design.ncols(),
            );
            crate::inner::pirls::score_log_det_h(
                &family,
                &self.y,
                &fit.eta,
                &fit.mu,
                &prior_w,
                &self.x_design,
                &s_total,
                fit,
            )
        } else {
            fit.log_det_a()
        };
        let ls_sum: f64 = self
            .y
            .iter()
            .map(|&yi| family.loss.saturated_log_lik(yi, phi))
            .sum();
        crate::score::reml_score_from_parts(dp, phi, self.mp, log_det_h, log_det_lambda_s, ls_sum)
    }
}
