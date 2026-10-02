//! `EnvelopeScore` — 1-D outer Newton REML/LAML closed-form score,
//! no shape parameters.
//!
//! Important identity used throughout: since `A := H = X'WX + λS`,
//!
//! ```text
//!   tr(H⁻¹ X'WX) = tr(H⁻¹ (A - λS)) = p - λ·tr(H⁻¹ S)
//! ```
//!
//! So we never need to form `X'WX` separately — one `tr(H⁻¹ S)` solve
//! gives both quantities. This is what lets PIRLS-iterative families
//! (where W changes each iteration) share the Gaussian score body
//! without re-allocating.
//!
//! `S: LinearSolver` (default `CholeskySolver`) flows through the inner
//! fit — the score body calls `inner_fit.log_det_a()` and
//! `inner_fit.trace_a_inv(...)` rather than touching the factor directly.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::Gaussian;
use crate::inner::{CholeskySolver, GaussianClosedFormInner, GaussianInnerFit, LinearSolver};
use crate::traits::{CoordsKind, InnerSolver, Loss, ScoreDerivatives};

use super::profile::{MgcvTwoSigmaProfile, Profile};

/// Closed-form REML / LAML envelope score, generic over the loss `L`,
/// the inner solver `I`, the dispersion profile `P`, and the linear
/// backend `S`. `K` and `V` are erased — the score only needs `Loss`
/// (for `saturated_log_lik`) and the converged inner fit's
/// β / working_weights / deviance. `P` names the σ² convention (see
/// `Profile` trait above).
///
/// This is the trait architecture's keystone: a SINGLE score impl serves
/// every family that fits in the `RemlScoreParts` mould. Plan §3.4 calls
/// this the `ClosedFormEnvelope<F, B>` impl; we name it `EnvelopeScore`.
pub struct EnvelopeScore<
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver = CholeskySolver,
> {
    pub inner: I,
    pub loss: L,
    pub profile: P,
    /// Per-term penalty list `Vec<S_j>` in current basis coords. Single-
    /// smooth fits use `s_list.len() == 1`; multi-smooth `Additive` fits
    /// use one block per smoothing parameter.
    pub s_list: Vec<Array2<f64>>,
    /// Per-term ranks. The score's `log|λS|+` term is
    /// `Σ_j (rank_j · ρ_j + log_pseudo_det_j)`.
    pub rank_s_list: Vec<usize>,
    pub mp: usize,
    /// Per-term `log|S_j|+` at `λ_j = 1`.
    pub log_pseudo_det_s_list: Vec<f64>,
    pub coords: CoordsKind,
    /// Raw response `y` — needed for `Σ ls(y_i)` in the score formula.
    /// (`inner.y` is also available but we keep our own copy so the score
    /// doesn't have to peek inside the InnerSolver.)
    pub y: Array1<f64>,
    pub _solver: PhantomData<S>,
    /// Cell-based diagnostic counters. Bumped by outer.rs line-search and
    /// by `value_grad_hess`'s inner-fit call. Read after fit via
    /// `score.stats().unwrap().snapshot()`.
    pub stats: crate::stats::FitStats,
    /// Last-accepted (β, b1, λ) for IFT warm-start in subsequent
    /// `value()` / `value_grad_hess()` calls. Mirrors
    /// `ShapeAwareEnvelopeScore::accepted_state`. Populated after every
    /// successful `value_grad_hess`; consumed by `ift_propagated_beta`.
    /// Only meaningful for families that opt in via `Loss::allows_no_refresh`.
    pub accepted_state: std::cell::RefCell<Option<EnvelopeAcceptedState>>,
}

/// IFT-propagation state cached by `EnvelopeScore` for the NoRefresh
/// line-search shortcut. Same shape as
/// `super::shape_aware::score::AcceptedState` minus the shape axis.
#[derive(Clone)]
pub struct EnvelopeAcceptedState {
    /// Converged β at the last accepted point.
    pub beta: Array1<f64>,
    /// First-order IFT derivative `b1[:, k] = ∂β/∂ρ_k = -λ_k · A⁻¹ · S_k · β`
    /// at the last accepted point. Shape (p, n_terms).
    pub b1: Array2<f64>,
    /// λ vector at the last accepted point. Length n_terms.
    pub lambda: Vec<f64>,
}

/// Phase-0 / Phase-1 convenience type alias for the Gaussian one-Cholesky
/// inner with the mgcv two-σ² convention. PIRLS-iterative families wire
/// `EnvelopeScore<L, PirlsInner<L, K, V>, P>` directly.
pub type GaussianClosedFormScore = EnvelopeScore<
    Gaussian,
    GaussianClosedFormInner<CholeskySolver>,
    MgcvTwoSigmaProfile,
    CholeskySolver,
>;

impl<L, I, P, S> EnvelopeScore<L, I, P, S>
where
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Generic constructor — accepts any `(Loss, InnerSolver, Profile)`
    /// triple. Used by PIRLS-based fit entry points (`fit_binomial_cr`,
    /// `fit_poisson_cr`, …) for a consistent build pattern across families.
    pub fn with_inner(
        inner: I,
        loss: L,
        profile: P,
        y: Array1<f64>,
        s_list: Vec<Array2<f64>>,
        rank_s_list: Vec<usize>,
        mp: usize,
        log_pseudo_det_s_list: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(s_list.len(), rank_s_list.len());
        debug_assert_eq!(s_list.len(), log_pseudo_det_s_list.len());
        Self {
            inner,
            loss,
            profile,
            s_list,
            rank_s_list,
            mp,
            log_pseudo_det_s_list,
            coords: CoordsKind::Identity,
            y,
            _solver: PhantomData,
            stats: crate::stats::FitStats::new(),
            accepted_state: std::cell::RefCell::new(None),
        }
    }
}

impl GaussianClosedFormScore {
    /// Phase-0 convenience constructor — wires the closed-form Gaussian
    /// inner with the mgcv two-σ² profile. Equivalent to
    /// `with_inner(GaussianClosedFormInner::new(...), Gaussian, MgcvTwoSigmaProfile, ...)`.
    pub fn new(
        x_design: Array2<f64>,
        y: Array1<f64>,
        s_list: Vec<Array2<f64>>,
        weights: Option<Array1<f64>>,
        rank_s_list: Vec<usize>,
        mp: usize,
        log_pseudo_det_s_list: Vec<f64>,
    ) -> Self {
        let inner = GaussianClosedFormInner::<CholeskySolver>::new(
            x_design,
            y.clone(),
            weights,
            s_list.clone(),
        );
        Self::with_inner(
            inner,
            Gaussian,
            MgcvTwoSigmaProfile,
            y,
            s_list,
            rank_s_list,
            mp,
            log_pseudo_det_s_list,
        )
    }
}

impl<L, I, P, S> ScoreDerivatives for EnvelopeScore<L, I, P, S>
where
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver,
{
    fn dim(&self) -> usize {
        self.s_list.len()
    }

    fn coords(&self) -> CoordsKind {
        self.coords.clone()
    }

    fn value(&self, theta: &Array1<f64>) -> Result<f64> {
        // Cheap value path: skip the analytic Hessian work, and use IFT
        // warm-start if accepted state cached + family eligible (Wood
        // 2011 Phase 5 / mgcv_rust `fit_pirls_cached:1077-1094`). For
        // Gamma/Poisson/Bernoulli/Binomial this typically shaves 2-5
        // PIRLS iters per trial λ.
        let beta_warm = self.ift_propagated_beta(theta);
        let inner: GaussianInnerFit<S> = match &beta_warm {
            Some(b) => {
                self.stats.bump_no_refresh_attempt();
                let fit = self.inner.fit_warm(theta, Some(b))?;
                self.stats.bump_no_refresh_hit();
                fit
            }
            None => self.inner.fit(theta)?,
        };
        self.stats.record_pirls_call(inner.iterations);
        let (v, _, _) = self.compute_value_grad_from_fit(theta, &inner)?;
        Ok(v)
    }

    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        let (v, g, _) = self.value_grad_hess(theta)?;
        Ok((v, g))
    }

    /// ρ axis bounds for the outer Newton. λ = exp(ρ); ±30 corresponds to
    /// λ ∈ [≈1e-13, ≈1e13] — the saturating-λ regime mgcv calls
    /// "effectively penalised to zero" / "effectively unpenalised". Without
    /// this clamp the outer Newton can walk a saturating-λ ridge to
    /// numerically-infinite λ (ρ ≈ 700 with the FD Hessian on the ti
    /// 3-margin smoke fixture), then "converge" on the relative-to-score
    /// criterion `|g| < 1e-3·(|v|+1)` purely because `|v|` grew during the
    /// drift. The analytic Hessian takes accurate enough steps that this
    /// accidental criterion no longer fires — so we clamp ρ to a physically
    /// meaningful box, matching mgcv's `optim.scale.bound`.
    fn axis_bounds(&self) -> Option<Vec<(f64, f64)>> {
        Some(vec![(-30.0, 30.0); self.s_list.len()])
    }

    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        // Run the (expensive) inner fit ONCE and reuse it for value/grad and
        // the analytic Hessian — the whole point of the analytic path is to
        // avoid the `1 + 2d` inner fits the FD Hessian needs. IFT warm-start
        // when accepted state is cached + family eligible.
        let beta_warm = self.ift_propagated_beta(theta);
        let inner: GaussianInnerFit<S> = match &beta_warm {
            Some(b) => self.inner.fit_warm(theta, Some(b))?,
            None => self.inner.fit(theta)?,
        };
        self.stats.record_pirls_call(inner.iterations);
        let (v, g, hin) = self.compute_value_grad_from_fit(theta, &inner)?;
        let hess = match self.hess_analytic(&inner, &hin) {
            Some(h) => h,
            None => self.hess_via_fd(theta)?,
        };

        // Stash NoRefresh accepted state for the next line-search trial.
        // b1[:,k] = -λ_k · A⁻¹ · S_k · β reuses `hin.a_inv` (already
        // materialised) — m × O(p²) GEMVs, negligible vs the inner fit.
        // Mirrors `ShapeAwareEnvelopeScore::compute_value_grad_hess_rho_only_with_fit`
        // and mgcv_rust `gam_optimized.rs:1574-1593` (warm_state write).
        if self.loss.allows_no_refresh() && hin.a_inv.nrows() == inner.p {
            let n_terms = self.s_list.len();
            let p_dim = inner.p;
            let mut b1 = Array2::<f64>::zeros((p_dim, n_terms));
            for k in 0..n_terms {
                let s_k_beta: Array1<f64> = self.s_list[k].dot(&inner.beta);
                let ainv_sk_beta: Array1<f64> = hin.a_inv.dot(&s_k_beta);
                let lam_k = hin.lambda_j[k];
                for r in 0..p_dim {
                    b1[[r, k]] = -lam_k * ainv_sk_beta[r];
                }
            }
            *self.accepted_state.borrow_mut() = Some(EnvelopeAcceptedState {
                beta: inner.beta.clone(),
                b1,
                lambda: hin.lambda_j.clone(),
            });
        }

        Ok((v, g, hess))
    }

    fn stats(&self) -> Option<&crate::stats::FitStats> {
        Some(&self.stats)
    }
}

/// Cached quantities from the converged inner fit needed to assemble the
/// analytic outer-Newton Hessian — gathered once by
/// [`EnvelopeScore::compute_value_grad_from_fit`] so [`EnvelopeScore::hess_analytic`]
/// never re-derives them (and never re-runs the inner fit).
pub(crate) struct HessInputs {
    /// λ_j = exp(ρ_j) per term.
    lambda_j: Vec<f64>,
    /// β'S_jβ per term.
    bsb_per_term: Vec<f64>,
    /// tr(A⁻¹ S_j) per term (Fisher A).
    tr_hinv_s_per_term: Vec<f64>,
    /// Score-side σ̂² at this θ.
    sigma2: f64,
    /// `∂σ²/∂ρ_i` per term, or `None` when no closed form exists (→ FD).
    dsigma2_drho: Option<Vec<f64>>,
    /// Whether the gradient carried a Tk·KK' (non-canonical-link) term — the
    /// analytic Hessian does not differentiate that yet, so we fall back.
    has_tk_kkt: bool,
    /// Dense `A⁻¹` (p × p), materialised once in
    /// [`EnvelopeScore::compute_value_grad_from_fit`] and forwarded to
    /// [`EnvelopeScore::hess_analytic`] so neither layer re-inverts. Empty
    /// `(0,0)` when the score path didn't need it (Tk·KK' fallback to FD).
    a_inv: Array2<f64>,
}

impl<L, I, P, S> EnvelopeScore<L, I, P, S>
where
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Coupled `(value, grad)` — the actual closed-form work. Runs the inner
    /// fit, then delegates the formula to [`Self::compute_value_grad_from_fit`].
    pub(crate) fn compute_value_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        let inner: GaussianInnerFit<S> = self.inner.fit(theta)?;
        let (v, g, _) = self.compute_value_grad_from_fit(theta, &inner)?;
        Ok((v, g))
    }

    /// The closed-form value + gradient on an ALREADY-converged inner fit,
    /// plus the cached [`HessInputs`] the analytic Hessian needs. Splitting
    /// the inner-fit call out lets `value_grad_hess` reuse one fit for both
    /// the gradient and the Hessian.
    pub(crate) fn compute_value_grad_from_fit(
        &self,
        theta: &Array1<f64>,
        inner: &GaussianInnerFit<S>,
    ) -> Result<(f64, Array1<f64>, HessInputs)> {
        debug_assert_eq!(
            theta.len(),
            self.s_list.len(),
            "EnvelopeScore: theta length {} must equal s_list length {}",
            theta.len(),
            self.s_list.len()
        );
        let n_terms = self.s_list.len();
        // λ_j = exp(ρ_j) per term.
        let lambda_j: Vec<f64> = theta.iter().map(|&r| r.exp()).collect();

        // Per-term β' S_j β and tr(H⁻¹ S_j). The combined `λ S β` becomes
        // `Σ_j λ_j (S_j β)`; the score uses these per-term to compute the
        // multi-d gradient `∂REML/∂ρ_j = λ_j β'S_j β/(2σ²) + λ_j tr(H⁻¹ S_j)/2
        // - rank_j/2`. v0.x's `reml_gradient_multi_*` follows the same shape.
        //
        // Perf: materialise `A⁻¹` ONCE here and reuse for every per-term
        // trace, instead of letting each `inner.trace_a_inv(s_j)` re-invert
        // (the trait's default path: `let a_inv = Self::invert(fact); ...`).
        // With m=10 smooths this saved 9 redundant O(p³) inversions per
        // outer probe — the 10-D Gaussian fit gap. mgcv_rust does the same
        // (`reml_gradient_mgcv_exact_closed_form_inner` caches `system.a_inv`).
        let a_inv = inner.a_inv();
        let p_dim = inner.p;
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut tr_hinv_s_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut bsb_total = 0.0_f64; // Σ_j λ_j β'S_jβ — used in dp / σ²-eq
        let mut tr_hinv_s_lambda_total = 0.0_f64; // Σ_j λ_j tr(H⁻¹S_j)
        for j in 0..n_terms {
            let s_j = &self.s_list[j];
            let s_beta = s_j.dot(&inner.beta);
            let bsb_j: f64 = inner
                .beta
                .iter()
                .zip(s_beta.iter())
                .map(|(a, b)| a * b)
                .sum();
            // tr(A⁻¹·S_j) = Σ_{i,k} A⁻¹[i,k] · S_j[k,i]  — v0.x's iteration
            // order from `src/reml/mod.rs:914-918`, identical FP to
            // `LinearSolver::trace_a_inv` but without the per-call invert.
            let mut tr_hinv_s_j = 0.0_f64;
            for ii in 0..p_dim {
                for kk in 0..p_dim {
                    tr_hinv_s_j += a_inv[[ii, kk]] * s_j[[kk, ii]];
                }
            }
            bsb_per_term.push(bsb_j);
            tr_hinv_s_per_term.push(tr_hinv_s_j);
            bsb_total += lambda_j[j] * bsb_j;
            tr_hinv_s_lambda_total += lambda_j[j] * tr_hinv_s_j;
        }
        // Tk·KK' Newton path (non-canonical-link): per-term Newton-A traces
        // when present, otherwise Fisher per-term. Used per-k below.

        // Identity: tr(H⁻¹ X'WX) = p - Σ_j λ_j·tr(H⁻¹ S_j). Fisher
        // version for the σ²-grad denominator (mgcv convention).
        let tr_hinv_xtwx = (inner.p as f64) - tr_hinv_s_lambda_total;

        // `log|H|` — defaults to the backend's `log|A|` off the Fisher
        // factor; lazily compute the Newton-W `Σ log|λ_i|` via the
        // `InnerSolver::lazy_newton_log_det_h` trait method when the loss
        // opts into the Newton path (non-canonical InverseGaussian + log)
        // so we match v0.x `src/reml/mod.rs:436-459`. `tr(H⁻¹S)` continues
        // to use Fisher H per v0.x's `system.tr_a` convention (the
        // score's `(n − tr_a)` denominator expects Fisher).
        //
        // **Lazy computation pattern** (mgcv_rust port): the Newton-A
        // pieces used to be materialised inside `PirlsInner::fit` (eigh
        // per inner fit — O(p³) regression on NegBin bench). Moved out
        // per `src/pirls/mod.rs::fit_pirls_cached` shape — the score
        // body computes them on demand here, once per probe.
        let log_det_h_fisher = inner.log_det_a();
        let log_det_h = self
            .inner
            .lazy_newton_log_det_h(inner, theta)
            .unwrap_or(log_det_h_fisher);
        // log|λS|+ for multi-smooth: Σ_j (rank_j · ρ_j + log_pseudo_det_j).
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            log_det_lambda_s +=
                (self.rank_s_list[j] as f64) * theta[j] + self.log_pseudo_det_s_list[j];
        }

        // σ² dispatch via the `Profile` type parameter — type-level choice,
        // not a runtime branch (architecture-assumptions.md §D3).
        //
        // Multi-smooth lift: the Profile signature takes a scalar `lambda`
        // and scalar `bsb`; here we pass `lambda = 1` and the pre-computed
        // `bsb_total = Σ_j λ_j β'S_jβ` so `dp = D + 1·bsb_total` matches
        // the multi-smooth Dp.
        let score_sigma2 =
            match self
                .profile
                .dispersion(&self.loss, inner, 1.0, bsb_total, tr_hinv_xtwx, self.mp)
            {
                Some(phi) => phi,
                None => {
                    // Unphysical probe — sentinel score, zero grad, and a
                    // benign HessInputs that makes `hess_analytic` bail to FD.
                    let hin = HessInputs {
                        lambda_j: lambda_j.clone(),
                        bsb_per_term: vec![0.0; n_terms],
                        tr_hinv_s_per_term: vec![0.0; n_terms],
                        sigma2: 1.0,
                        dsigma2_drho: None,
                        has_tk_kkt: false,
                        a_inv: Array2::<f64>::zeros((0, 0)),
                    };
                    return Ok((1e12, Array1::zeros(n_terms), hin));
                }
            };

        // `Dp = D + Σ_j λ_j·β'S_jβ`.
        let dp = inner.deviance + bsb_total;

        // Σ ls(y_i; σ²) — saturated log-likelihood sum at the score's σ².
        // Gamma/InverseGaussian's sat_lik depend on σ² via the φ-term that
        // Phase-2b v0.2 port reinstated; other families ignore the scale arg.
        let ls_sum: f64 = self
            .y
            .iter()
            .map(|&yi| self.loss.saturated_log_lik(yi, score_sigma2))
            .sum();

        // GamFit3 form (mgcv `gam.fit3.r:616-617`, v0.x
        // `src/pirls/mod.rs::ScoreFormula::GamFit3`):
        //
        //   REML = Dp/(2σ²) - Mp/2·log(2πσ²) + log|H|/2 - log|λS|+/2 - Σ ls
        //
        // Phase-2b v0.2 port (2026-05-24): switched from the Gaussian-
        // shortcut `+(n-Mp)/2·log(2πσ²) - Σls=0` form to the general form.
        // The two coincide for σ²-independent sat_lik (Gaussian's
        // `-n/2·log(2πφ)` exactly fills the `+n/2·log(2πφ)` gap between
        // `(n-Mp)/2` and `-Mp/2`); for σ²-DEPENDENT sat_lik (Gamma,
        // InverseGaussian) the general form is the only correct one.
        let two_pi = 2.0 * std::f64::consts::PI;
        let mp_f = self.mp as f64;
        let reml = dp / (2.0 * score_sigma2) - 0.5 * mp_f * (two_pi * score_sigma2).ln()
            + 0.5 * log_det_h
            - 0.5 * log_det_lambda_s
            - ls_sum;

        // Per-term gradient: ∂REML/∂ρ_j = λ_j β'S_jβ/(2σ²) + λ_j tr(H⁻¹S_j)/2
        //   - rank_j/2  (mgcv `reml_gradient_multi_*` shape).
        let mut g = Array1::<f64>::zeros(n_terms);
        for j in 0..n_terms {
            g[j] = lambda_j[j] * bsb_per_term[j] / (2.0 * score_sigma2)
                + 0.5 * lambda_j[j] * tr_hinv_s_per_term[j]
                - 0.5 * (self.rank_s_list[j] as f64);
        }

        // Tk·KK' contribution for non-canonical-link families. The W matrix
        // used in `log|H|` depends on β (through μ), so `d(log|H|)/dρ_k`
        // carries the explicit Tk·KK' term `Σᵢ a1[i] · η₁_k[i] · lev_uw[i]`
        // — port of mgcv_rust `reml_gradient_mgcv_exact_ift_newton_at_beta`
        // (`src/reml/mod.rs:2480-2483`). For multi-smooth, η₁_k is per-term
        // (PIRLS supplies `eta1_per_term[k]`). The trace contribution also
        // switches to the per-term Newton-A trace so the gradient stays
        // internally consistent (uses one A everywhere).
        let lazy_tk_kkt = self.inner.lazy_tk_kkt_inputs(inner, theta);
        let has_tk_kkt = lazy_tk_kkt.is_some();
        // Fisher-weight families: `W` still depends on β through μ (Poisson:
        // W = μ), so `log|A|` has a β-chain even with a canonical link. The
        // analytic Hessian below stays the envelope one; only the gradient,
        // which decides where the optimiser stops, needs the exact term.
        if !has_tk_kkt {
            if let Some(chain) = self.inner.fisher_log_det_beta_chain(inner, theta) {
                for k in 0..n_terms {
                    g[k] += 0.5 * chain[k];
                }
            }
        }
        if let Some(ref tk) = lazy_tk_kkt {
            debug_assert_eq!(
                tk.eta1_per_term.len(),
                n_terms,
                "EnvelopeScore: eta1_per_term length {} must equal n_terms {}",
                tk.eta1_per_term.len(),
                n_terms
            );
            debug_assert_eq!(
                tk.tr_a_newton_inv_s_per_term.len(),
                n_terms,
                "EnvelopeScore: tr_a_newton_inv_s_per_term length {} must equal n_terms {}",
                tk.tr_a_newton_inv_s_per_term.len(),
                n_terms
            );
            let n = inner.n;
            for k in 0..n_terms {
                let mut tk_kkt = 0.0_f64;
                let eta1_k = &tk.eta1_per_term[k];
                for i in 0..n {
                    tk_kkt += tk.a1[i] * eta1_k[i] * tk.lev_uw[i];
                }
                // Rewrite this term's gradient to use the Newton-A trace +
                // Tk·KK' β-chain contribution (vs the Fisher-only branch
                // computed above) — mirrors mgcv_rust:2482-2483.
                g[k] = lambda_j[k] * bsb_per_term[k] / (2.0 * score_sigma2)
                    + 0.5 * lambda_j[k] * tk.tr_a_newton_inv_s_per_term[k]
                    - 0.5 * (self.rank_s_list[k] as f64)
                    + 0.5 * tk_kkt;
            }
        }

        // Analytic-Hessian chain term `∂σ²/∂ρ` (closed-form profiles only).
        let dsigma2_drho = self.profile.dispersion_drho(
            &self.loss,
            inner,
            score_sigma2,
            &bsb_per_term,
            &lambda_j,
            self.mp,
        );

        let hess_inputs = HessInputs {
            lambda_j,
            bsb_per_term,
            tr_hinv_s_per_term,
            sigma2: score_sigma2,
            dsigma2_drho,
            has_tk_kkt,
            a_inv,
        };

        Ok((reml, g, hess_inputs))
    }

    /// Analytic outer-Newton Hessian `H[i,j] = ∂g[j]/∂ρ_i`, assembled from
    /// the converged inner fit's cached quantities — NO extra inner fits.
    ///
    /// Ported from mgcv_rust's analytic REML Hessian
    /// (`nn_exploring/src/reml/mod.rs::reml_hessian_multi_qr` and
    /// `reml/fastreml.rs::compute_d_det_xxs`). The gamrs gradient is the
    /// frozen-σ² envelope form
    ///
    /// ```text
    ///   g[j] = λ_j·β'S_jβ /(2σ²) + ½·λ_j·tr(A⁻¹S_j) − ½·rank_j
    /// ```
    ///
    /// so differentiating wrt ρ_i (β via the IFT `dβ/dρ_i = −λ_i A⁻¹ S_i β`,
    /// W frozen at convergence — the Gaussian / observed-info Hessian mgcv
    /// uses for the profiled fREML score) gives three pieces:
    ///
    /// 1. **Trace (`log|X'WX+S|`) curvature** — mgcv `compute_d_det_xxs`'s
    ///    `dxxs_d2`:
    ///    `½·[δ_ij·λ_j·tr(A⁻¹S_j) − λ_iλ_j·tr(A⁻¹S_i A⁻¹S_j)]`.
    /// 2. **Data-fit at fixed σ²** — `∂(λ_j β'S_jβ)/∂ρ_i / (2σ²)` with
    ///    `∂(λ_j β'S_jβ)/∂ρ_i = δ_ij·λ_j·β'S_jβ − 2λ_iλ_j·β'S_i A⁻¹ S_jβ`.
    ///    This is the surviving `bSb2` content of mgcv `Sl.iftChol`
    ///    specialised to the envelope (the `rss2` cross-terms fold into the
    ///    σ² chain below for the profiled families and vanish for σ²≡1).
    /// 3. **σ² chain** — `−(λ_j β'S_jβ /(2σ⁴))·∂σ²/∂ρ_i` (zero for σ²≡1).
    ///
    /// The `log|λS|+` term is linear in ρ so its second derivative is 0
    /// (mgcv `ldet_s_d2 = 0` for singleton penalties).
    ///
    /// Returns `None` when the gradient carried a Tk·KK' non-canonical-link
    /// term (not yet differentiated analytically) or when `∂σ²/∂ρ` has no
    /// closed form — the caller then keeps the FD Hessian.
    pub(crate) fn hess_analytic(
        &self,
        inner: &GaussianInnerFit<S>,
        hin: &HessInputs,
    ) -> Option<Array2<f64>> {
        // Non-canonical Tk·KK' gradient is not differentiated here.
        if hin.has_tk_kkt {
            return None;
        }
        // No closed-form σ² derivative (e.g. Gamma) → FD fallback.
        let dsigma2 = hin.dsigma2_drho.as_ref()?;

        let m = self.s_list.len();
        let sigma2 = hin.sigma2;
        let lambda = &hin.lambda_j;
        let bsb = &hin.bsb_per_term;
        let tr_per = &hin.tr_hinv_s_per_term;

        // A⁻¹ once (dense) — reused for every `A⁻¹ S_j β` and the
        // `tr(A⁻¹ S_i A⁻¹ S_j)` products. Re-uses the inversion already
        // materialised by `compute_value_grad_from_fit` (stored in
        // `hin.a_inv`) so the analytic Hessian's linear-algebra cost is one
        // O(p³) inversion total, not two. Borrows when the cached version
        // is usable to avoid an O(p²) clone per outer iter.
        let a_inv_owned: Array2<f64>;
        let a_inv: &Array2<f64> = if hin.a_inv.nrows() == inner.p {
            &hin.a_inv
        } else {
            a_inv_owned = inner.a_inv();
            &a_inv_owned
        };

        // Per-term: A⁻¹ S_j β  (the dβ/dρ_j direction without the −λ_j) and
        // S_j β.
        let mut s_beta: Vec<Array1<f64>> = Vec::with_capacity(m);
        let mut ainv_s_beta: Vec<Array1<f64>> = Vec::with_capacity(m);
        for j in 0..m {
            let sjb = self.s_list[j].dot(&inner.beta);
            let ainv_sjb = a_inv.dot(&sjb);
            s_beta.push(sjb);
            ainv_s_beta.push(ainv_sjb);
        }

        // Per-term A⁻¹ S_j (dense p×p) for the trace product `tr(A⁻¹S_i A⁻¹S_j)`.
        let mut ainv_s: Vec<Array2<f64>> = Vec::with_capacity(m);
        for j in 0..m {
            ainv_s.push(a_inv.dot(&self.s_list[j]));
        }

        // === W-chain prep (GLM / quantile families only) ===
        // `dW/dρ_i = diag(dw_deta · (X·dβ/dρ_i))`, `dβ/dρ_i = −λ_i A⁻¹S_iβ`.
        // The trace term picks up `−tr(A⁻¹·X'(dW_i)X·A⁻¹·S_j)
        //   = −Σ_k dW_i[k]·(X·A⁻¹S_jA⁻¹·X')_{kk}`.
        // We precompute, per term, the n-vectors:
        //   xdbeta[i][k] = (X·dβ/dρ_i)_k                 (so dW_i = dw_deta⊙xdbeta[i])
        //   lev_mj[j][k] = (X·A⁻¹S_jA⁻¹·X')_{kk}         (leverage-like)
        //
        // Gated on `loss.skip_w_chain_in_hessian()`: Gamma opts out per
        // `nn_exploring/src/smooth.rs:48-53` ("keep paired derivatives on
        // the consistent closed-form path") because its working-response
        // REML treats W as β-frozen, so the analytic Hessian must too.
        let skip_w_chain = self.loss.skip_w_chain_in_hessian();
        let w_chain: Option<(Vec<Array1<f64>>, Vec<Array1<f64>>)> = match (
            inner.dw_deta.as_ref(),
            inner.x_design.as_ref(),
            skip_w_chain,
        ) {
            (Some(_dw), Some(x), false) => {
                let n = x.nrows();
                // xdbeta[i] = X·(−λ_i A⁻¹S_iβ) = −λ_i · X·ainv_s_beta[i]
                let mut xdbeta: Vec<Array1<f64>> = Vec::with_capacity(m);
                for i in 0..m {
                    let mut v = x.dot(&ainv_s_beta[i]);
                    v.mapv_inplace(|t| -lambda[i] * t);
                    xdbeta.push(v);
                }
                // lev_mj[j][k] = Σ_b (X·ainv_s[j]·A⁻¹)[k,b]·X[k,b]
                //              = row_k(X·ainv_s[j]·A⁻¹) · row_k(X)
                let mut lev_mj: Vec<Array1<f64>> = Vec::with_capacity(m);
                for j in 0..m {
                    let xmj = x.dot(&ainv_s[j]).dot(a_inv); // (n×p)
                    let mut diag = Array1::<f64>::zeros(n);
                    for k in 0..n {
                        let mut s = 0.0;
                        for b in 0..x.ncols() {
                            s += xmj[[k, b]] * x[[k, b]];
                        }
                        diag[k] = s;
                    }
                    lev_mj.push(diag);
                }
                Some((xdbeta, lev_mj))
            }
            _ => None,
        };

        // The W-chain Hessian contribution below reads `inner.dw_deta`; if
        // we skipped the W-chain prep, hide `dw_deta` so the contribution
        // also evaluates to zero (avoids assembling a half-formed chain).
        let dw_deta_for_hess: Option<&Array1<f64>> = if skip_w_chain {
            None
        } else {
            inner.dw_deta.as_ref()
        };

        let mut h = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in i..m {
                // --- 1) trace curvature (dxxs_d2) ---
                // tr(A⁻¹S_i A⁻¹S_j) = Σ (A⁻¹S_i) ⊙ (A⁻¹S_j)ᵀ
                let mut tr_prod = 0.0_f64;
                let ai = &ainv_s[i];
                let aj = &ainv_s[j];
                let p = ai.nrows();
                for r in 0..p {
                    for c in 0..p {
                        tr_prod += ai[[r, c]] * aj[[c, r]];
                    }
                }
                let mut term_trace = -lambda[i] * lambda[j] * tr_prod;
                if i == j {
                    term_trace += lambda[i] * tr_per[i];
                }
                // W-chain contribution to the trace term. The gradient's
                // trace piece is `g_tr[j] = ½·λ_j·tr(A⁻¹S_j)`, so its raw
                // ρ_i-derivative through W is
                //   ∂g_tr[j]/∂ρ_i = ½·λ_j·(−Σ_k dW_i[k]·lev_mj[j][k]),
                // with dW_i[k] = dw[k]·xdbeta[i][k]. The exact Hessian is
                // symmetric; the envelope form is not term-by-term, so we
                // symmetrise H[i,j] = ½(∂g_tr[j]/∂ρ_i + ∂g_tr[i]/∂ρ_j) to
                // match the FD reference (which symmetrises too).
                let mut term_w = 0.0_f64;
                if let (Some((xdbeta, lev_mj)), Some(dw)) = (w_chain.as_ref(), dw_deta_for_hess) {
                    let mut s_ij = 0.0; // Σ_k dW_i[k]·lev_mj[j][k]
                    let mut s_ji = 0.0; // Σ_k dW_j[k]·lev_mj[i][k]
                    let n = dw.len();
                    for k in 0..n {
                        s_ij += dw[k] * xdbeta[i][k] * lev_mj[j][k];
                        s_ji += dw[k] * xdbeta[j][k] * lev_mj[i][k];
                    }
                    let dgtr_j_drho_i = -0.5 * lambda[j] * s_ij;
                    let dgtr_i_drho_j = -0.5 * lambda[i] * s_ji;
                    term_w = 0.5 * (dgtr_j_drho_i + dgtr_i_drho_j);
                }
                let term1 = 0.5 * term_trace + term_w;

                // --- 2) data-fit at fixed σ² ---
                // ∂(λ_j β'S_jβ)/∂ρ_i = δ_ij·λ_j·β'S_jβ
                //                      − 2 λ_iλ_j · β'S_i A⁻¹ S_jβ
                let bsi_ainv_sj: f64 = s_beta[i].dot(&ainv_s_beta[j]);
                let mut d_bj = -2.0 * lambda[i] * lambda[j] * bsi_ainv_sj;
                if i == j {
                    d_bj += lambda[j] * bsb[j];
                }
                let term2 = d_bj / (2.0 * sigma2);

                // --- 3) σ² chain ---
                // −(λ_j β'S_jβ /(2σ⁴)) · ∂σ²/∂ρ_i
                let term3 = -(lambda[j] * bsb[j]) / (2.0 * sigma2 * sigma2) * dsigma2[i];

                let val = term1 + term2 + term3;
                h[[i, j]] = val;
                if i != j {
                    h[[j, i]] = val;
                }
            }
        }

        Some(h)
    }

    /// IFT-propagate β from `accepted_state` to a trial θ via
    /// `β_trial = β_acc + Σ_k b1[:, k] · Δρ_k`. Returns `None` if the
    /// family is on the NoRefresh skip-list, no accepted state cached,
    /// or the propagated β goes non-finite. The result is consumed by
    /// `InnerSolver::fit_warm` so PIRLS starts from a near-converged β
    /// instead of `Loss::initial_mu` — typically saves 2-5 IRLS iters
    /// per trial λ. Port of mgcv_rust `gam_optimized.rs:1434-1472`.
    fn ift_propagated_beta(&self, theta: &Array1<f64>) -> Option<Array1<f64>> {
        if !self.loss.allows_no_refresh() {
            return None;
        }
        let n_terms = self.s_list.len();
        if theta.len() != n_terms {
            return None;
        }
        let state_ref = self.accepted_state.borrow();
        let state = state_ref.as_ref()?;
        if state.lambda.len() != n_terms {
            return None;
        }
        let p_dim = state.beta.len();
        let mut beta_warm = state.beta.clone();
        for k in 0..n_terms {
            let lam_trial = theta[k].exp().max(1e-300);
            let lam_saved = state.lambda[k].max(1e-300);
            let drho = (lam_trial / lam_saved).ln();
            if !drho.is_finite() {
                return None;
            }
            for r in 0..p_dim {
                beta_warm[r] += state.b1[[r, k]] * drho;
            }
        }
        if !beta_warm.iter().all(|x| x.is_finite()) {
            return None;
        }
        Some(beta_warm)
    }

    fn hess_via_fd(&self, theta: &Array1<f64>) -> Result<Array2<f64>> {
        let d = theta.len();
        let mut h = Array2::<f64>::zeros((d, d));
        let eps = 1.0e-5;
        for i in 0..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps;
            t_minus[i] -= eps;
            let (_, g_plus) = self.compute_value_grad(&t_plus)?;
            let (_, g_minus) = self.compute_value_grad(&t_minus)?;
            for j in 0..d {
                h[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
            }
        }
        for i in 0..d {
            for j in i + 1..d {
                let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
                h[[i, j]] = avg;
                h[[j, i]] = avg;
            }
        }
        Ok(h)
    }
}

// =============================================================================
// FD-match oracle: analytic Hessian (`hess_analytic`) must equal the
// correct-but-slow central-FD Hessian (`hess_via_fd`) on representative
// fixtures spanning Gaussian (closed-form inner) and PIRLS families
// (Poisson, ELF) plus a d>1 multi-smooth fit. The FD Hessian is the
// reference; the analytic one is the optimisation we ship.
// =============================================================================
#[cfg(test)]
mod fd_match_tests {
    use super::*;
    use crate::design::{Additive, Cr, DesignStrategy, MarginKind, PreparedDesign, TermSpec};
    use crate::family::{elf_identity, poisson_log, ElfLoss, Poisson};
    use crate::inner::{ArmijoElfInner, ArmijoElfOpts, PirlsInner, PirlsOpts};
    use crate::outer::{NewtonOpts, NewtonWithHalving};
    use crate::traits::OuterSolver;
    use ndarray::Array2;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures");
        p.push(format!("{name}.json"));
        p
    }

    /// Load `x_train` (n × d) and `y_train` from a fixture. Handles both
    /// `x_train: [[..],..]` (multi-col) and `x_train: [scalar,..]`.
    fn load_xy(name: &str) -> (Array2<f64>, Array1<f64>) {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture_path(name)).unwrap()).unwrap();
        let xj = &v["inputs"]["x_train"];
        let rows = xj.as_array().unwrap();
        let n = rows.len();
        let ncol = if rows[0].is_array() {
            rows[0].as_array().unwrap().len()
        } else {
            1
        };
        let mut x = Array2::<f64>::zeros((n, ncol));
        for (i, r) in rows.iter().enumerate() {
            if let Some(arr) = r.as_array() {
                for (j, c) in arr.iter().enumerate() {
                    x[[i, j]] = c.as_f64().unwrap();
                }
            } else {
                x[[i, 0]] = r.as_f64().unwrap();
            }
        }
        let y: Vec<f64> = v["inputs"]["y_train"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_f64().unwrap())
            .collect();
        (x, Array1::from_vec(y))
    }

    /// Max relative error between two Hessians (denominator floored at 1).
    fn max_rel(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let mut m = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let denom = y.abs().max(1.0);
            m = m.max((x - y).abs() / denom);
        }
        m
    }

    /// Assert `hess_analytic == hess_via_fd` at `theta` and at a perturbed
    /// `theta`. `bar` is the relative tolerance gate.
    fn assert_fd_match<L, I, P, S>(
        label: &str,
        score: &EnvelopeScore<L, I, P, S>,
        theta_hat: &Array1<f64>,
        bar: f64,
    ) where
        L: Loss,
        I: InnerSolver<Fit = GaussianInnerFit<S>>,
        P: Profile<L>,
        S: LinearSolver,
    {
        let d = theta_hat.len();
        let perturb = {
            let mut t = theta_hat.clone();
            for i in 0..d {
                t[i] += 0.37 + 0.11 * (i as f64);
            }
            t
        };
        for (tag, theta) in [("theta_hat", theta_hat), ("perturbed", &perturb)] {
            let inner = score.inner.fit(theta).unwrap();
            let (_, _, hin) = score.compute_value_grad_from_fit(theta, &inner).unwrap();
            let h_anal = score
                .hess_analytic(&inner, &hin)
                .unwrap_or_else(|| panic!("{label}/{tag}: hess_analytic returned None"));
            // The analytic Hessian differentiates the gradient's ENVELOPE part;
            // the Fisher β-chain of log|A| is in the gradient only (an FD
            // Hessian of the full gradient cost 2.3–3.7× the time for the same
            // iterations). So difference the gradient with that term removed.
            let envelope_grad = |t: &Array1<f64>| -> Array1<f64> {
                let (_, mut g) = score.value_and_grad(t).unwrap();
                let fit = score.inner.fit(t).unwrap();
                if let Some(chain) = score.inner.fisher_log_det_beta_chain(&fit, t) {
                    for k in 0..chain.len() {
                        g[k] -= 0.5 * chain[k];
                    }
                }
                g
            };
            let h = 1e-5;
            let mut h_fd = Array2::<f64>::zeros((d, d));
            for i in 0..d {
                let (mut tp, mut tm) = (theta.clone(), theta.clone());
                tp[i] += h;
                tm[i] -= h;
                let (gp, gm) = (envelope_grad(&tp), envelope_grad(&tm));
                for j in 0..d {
                    h_fd[[j, i]] = (gp[j] - gm[j]) / (2.0 * h);
                }
            }
            let h_fd = (&h_fd + &h_fd.t()) * 0.5;
            let rel = max_rel(&h_anal, &h_fd);
            eprintln!("[fd-match] {label}/{tag}: max_rel = {rel:.3e}  (d={d})");
            assert!(
                rel <= bar,
                "{label}/{tag}: analytic-vs-FD Hessian max_rel {rel:.3e} exceeds {bar:.1e}\n\
                 analytic = {h_anal:?}\n fd = {h_fd:?}",
            );
        }
    }

    #[test]
    fn fd_match_gaussian_closed_form_1d() {
        let (x, y) = load_xy("1d_gaussian_smooth_n500_k10_cr");
        let prep: PreparedDesign = Cr { k: 10 }.prepare(x.view()).unwrap();
        let score = GaussianClosedFormScore::new(
            prep.x_design.clone(),
            y.clone(),
            prep.s_list.clone(),
            None,
            prep.rank_s_list.clone(),
            prep.mp,
            prep.log_pseudo_det_s_list.clone(),
        );
        let outer = NewtonWithHalving::new(NewtonOpts::default());
        let fit = outer.minimize(&score, Array1::zeros(1)).unwrap();
        // Closed-form inner ⇒ analytic Hessian is near machine-exact
        // (measured ~2e-10). Bar set well inside the 1e-4 gate.
        assert_fd_match("gaussian_1d", &score, &fit.theta, 1e-6);
    }

    #[test]
    fn fd_match_gaussian_additive_2d() {
        let (x, y) = load_xy("2d_gaussian_additive_n500_k10_cr");
        let terms = vec![
            TermSpec::Cr {
                col: 0,
                k: 10,
                pc: None,
            },
            TermSpec::Cr {
                col: 1,
                k: 10,
                pc: None,
            },
        ];
        let prep = Additive { terms }.prepare(x.view()).unwrap();
        assert_eq!(prep.s_list.len(), 2, "expected d=2 multi-smooth");
        let score = GaussianClosedFormScore::new(
            prep.x_design.clone(),
            y.clone(),
            prep.s_list.clone(),
            None,
            prep.rank_s_list.clone(),
            prep.mp,
            prep.log_pseudo_det_s_list.clone(),
        );
        let outer = NewtonWithHalving::new(NewtonOpts::default());
        let fit = outer.minimize(&score, Array1::zeros(2)).unwrap();
        // Multi-smooth (d=2) closed-form inner; measured ~1.5e-9.
        assert_fd_match("gaussian_additive_2d", &score, &fit.theta, 1e-6);
        let _ = MarginKind::Cr; // keep import used if TermSpec layout changes
    }

    #[test]
    #[ignore]
    fn diag_ti_hess_grid() {
        // Reproduce the ti 3-margin Gaussian design and compare analytic vs
        // FD Hessian on a grid of θ to locate where they diverge.
        let n = 400;
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut x = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (x0, x1, x2) = (next(), next(), next());
            x[[i, 0]] = x0;
            x[[i, 1]] = x1;
            x[[i, 2]] = x2;
            let noise = (next() - 0.5) * 0.2;
            y[i] = (2.0 * std::f64::consts::PI * x0).sin()
                + (1.5 * x1 - 0.5).powi(2)
                + 0.8 * x0 * x2
                + 0.5 * x1 * x2
                + noise;
        }
        let terms = vec![TermSpec::Ti {
            cols: vec![0, 1, 2],
            k: vec![4, 4, 4],
            bs: vec![MarginKind::Cr, MarginKind::Cr, MarginKind::Cr],
        }];
        let prep = Additive { terms }.prepare(x.view()).unwrap();
        eprintln!(
            "ti d={} ranks={:?} mp={}",
            prep.s_list.len(),
            prep.rank_s_list,
            prep.mp
        );
        let score = GaussianClosedFormScore::new(
            prep.x_design.clone(),
            y.clone(),
            prep.s_list.clone(),
            None,
            prep.rank_s_list.clone(),
            prep.mp,
            prep.log_pseudo_det_s_list.clone(),
        );
        for probe in [
            [0.0, 0.0, 0.0],
            [2.0, -1.0, 3.0],
            [5.0, 5.0, 5.0],
            [-2.0, 4.0, 1.0],
        ] {
            let theta = Array1::from_vec(probe.to_vec());
            let inner = score.inner.fit(&theta).unwrap();
            let (_, _, hin) = score.compute_value_grad_from_fit(&theta, &inner).unwrap();
            if let Some(ha) = score.hess_analytic(&inner, &hin) {
                let hf = score.hess_via_fd(&theta).unwrap();
                let rel = max_rel(&ha, &hf);
                eprintln!("θ={:?}: max_rel={:.3e}", probe, rel);
            } else {
                eprintln!("θ={:?}: analytic Hess unavailable (FD fallback)", probe);
            }
        }
        // Run the outer Newton and report convergence.
        let outer = NewtonWithHalving::new(NewtonOpts::default());
        match outer.minimize(&score, Array1::zeros(3)) {
            Ok(f) => eprintln!(
                "OUTER: converged={} iters={} grad_norm={:.3e} theta={:?}",
                f.converged, f.iterations, f.grad_norm, f.theta
            ),
            Err(e) => eprintln!("OUTER ERR: {e:?}"),
        }
        for r in [100.0, 200.0, 354.9999, 355.0, 400.0, 709.0] {
            let theta = Array1::from_vec(vec![r, r, r]);
            if let Ok((v, g, _)) = score.value_grad_hess(&theta) {
                eprintln!(
                    "  trace θ={r}: v={v:.6e} |g|inf={:.4e}",
                    g.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
                );
            } else {
                eprintln!("  trace θ={r}: non-finite");
            }
        }
    }

    #[test]
    fn fd_match_poisson_pirls_1d() {
        let (x, y) = load_xy("1d_poisson_log_n300_k10_cr");
        let prep = Cr { k: 10 }.prepare(x.view()).unwrap();
        let pirls = PirlsInner::<Poisson, _, _, CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family: poisson_log(),
            opts: PirlsOpts::default(),
            _solver: PhantomData,
        };
        let score = EnvelopeScore::<Poisson, _, _, CholeskySolver>::with_inner(
            pirls,
            Poisson,
            crate::score::FixedAtOneProfile,
            y.clone(),
            prep.s_list.clone(),
            prep.rank_s_list.clone(),
            prep.mp,
            prep.log_pseudo_det_s_list.clone(),
        );
        let outer = NewtonWithHalving::new(NewtonOpts::default());
        let fit = outer.minimize(&score, Array1::zeros(1)).unwrap();
        // PIRLS inner with the analytic W-chain term ⇒ matches FD to
        // ~1e-10 (the W-derivative is FD'd inside the inner at 1e-6, so
        // the residual is the FD-of-FD noise floor, still far inside 1e-4).
        assert_fd_match("poisson_1d", &score, &fit.theta, 1e-6);
    }

    #[test]
    fn fd_match_gamma_pirls_1d() {
        use crate::family::{gamma_log, Gamma};
        use crate::score::MgcvTwoSigmaProfile;
        let (x, y) = load_xy("1d_gamma_log_n300_k10_cr");
        let prep = Cr { k: 10 }.prepare(x.view()).unwrap();
        let pirls = PirlsInner::<Gamma, _, _, CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family: gamma_log(),
            opts: PirlsOpts::default(),
            _solver: PhantomData,
        };
        let score = EnvelopeScore::<Gamma, _, _, CholeskySolver>::with_inner(
            pirls,
            Gamma,
            MgcvTwoSigmaProfile,
            y.clone(),
            prep.s_list.clone(),
            prep.rank_s_list.clone(),
            prep.mp,
            prep.log_pseudo_det_s_list.clone(),
        );
        let outer = NewtonWithHalving::new(NewtonOpts::default());
        let fit = outer.minimize(&score, Array1::zeros(1)).unwrap();
        // PIRLS inner + Newton-on-φ profile + IFT σ²-chain factor
        // (-1/F'(φ̂)). The envelope-form analytic Hessian and the FD Hessian
        // agree to ~1e-3 here (measured 7.75e-4 at θ̂, ~1e-3 at the perturbed
        // probe). The gap is the dW/dη FD noise (PIRLS computes dw_deta via
        // central FD at 1e-6) compounded by the trigamma evaluation in the
        // IFT factor. Newton converges in 6 outer iters (vs 5 for the
        // pre-IFT FD-Hessian fallback), still within the 0.4.3 perf budget.
        assert_fd_match("gamma_1d", &score, &fit.theta, 2e-3);
    }

    #[test]
    fn fd_match_elf_pirls_1d() {
        let (x, y) = load_xy("quantile_oos_hetero_n800_cr");
        let prep = Cr { k: 10 }.prepare(x.view()).unwrap();
        let tau = 0.5_f64;
        let inner = ArmijoElfInner::<CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family: elf_identity(tau, 1.0, 0.01),
            opts: ArmijoElfOpts::default(),
            beta_init: None,
            _solver: PhantomData,
        };
        let score = EnvelopeScore::<ElfLoss, _, _, CholeskySolver>::with_inner(
            inner,
            ElfLoss {
                tau,
                sigma: 1.0,
                lambda: 0.01,
            },
            crate::score::FixedAtOneProfile,
            y.clone(),
            prep.s_list.clone(),
            prep.rank_s_list.clone(),
            prep.mp,
            prep.log_pseudo_det_s_list.clone(),
        );
        // Tighter tolerances than default (5e-7 / 1e-7 = mgcv parity) so
        // the analytic-vs-FD Hessian comparison sees a near-zero gradient
        // at the optimum. Without this the residual ‖g‖ leaks O(g·h) into
        // the FD Hessian and widens the analytic vs FD gap to ~3e-3.
        let opts = NewtonOpts {
            grad_tol: 1e-9,
            reml_tol: 1e-10,
            ..NewtonOpts::default()
        };
        let outer = NewtonWithHalving::new(opts);
        let fit = outer.minimize(&score, Array1::zeros(1)).unwrap();
        // ELF Armijo inner: the `∂W/∂η` term is a central FD of the ELF
        // sigmoid weight, so the analytic Hessian carries a small FD noise
        // floor (measured ~2e-5 at the perturbed probe). Bar 1e-4 — inside
        // the gate, comfortably above the noise.
        assert_fd_match("elf_1d", &score, &fit.theta, 1e-4);
    }
}
