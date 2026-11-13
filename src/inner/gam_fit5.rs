//! gam.fit5-style joint β + working-weights solver for the ocat extended family.
//!
//! Ocat doesn't fit the standard `PirlsInner` mould because its working
//! weights are NOT `prior/(V(μ) · g'(μ)²)`: mgcv's `ocat` family computes
//! working weights and the working response directly from the deviance
//! derivatives `Dmu, Dmu2` (cumulative-logit math, see
//! `crate::family::OcatLoss::deviance_per_obs` and v0.x `src/ocat.rs`).
//!
//! Layout mirrors v0.x `src/pirls/mod.rs::fit_pirls_ocat`:
//!   - per-iter: build (w, z) from Dmu / Dmu2 at the current η
//!   - solve `(X'WX + λS) β = X'Wz` via the configured backend
//!   - step-halve toward β_old when pdev > pdev_old or non-finite
//!   - converge when max |β − β_old| < tol
//!
//! θ (the log-gap thresholds) is **external** to this inner — owned by
//! `OcatLoss.thresholds`, which the outer joint-Newton refreshes per
//! probe. The score side reads it via the same `ShapeAwareEnvelopeScore`
//! machinery scat/Tweedie/NegBin already use.
//!
//! `S: LinearSolver` (default `CholeskySolver`) propagates the backend
//! choice into the emitted `GaussianInnerFit<S>`.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::{Family, IdentityLink, OcatLoss, OcatVariance};
use crate::traits::{InnerSolver, Loss};

use super::{
    add_penalty, beta_sbeta, halve_until_valid, split::SplitDesign, CholeskySolver, GaussianInnerFit,
    LinearSolver, PirlsOpts,
};

/// PIRLS-style inner solver for the `ocat` extended family. β is solved
/// jointly with the working weights at fixed thresholds `θ` (read from
/// `OcatLoss.thresholds`). Returns a `GaussianInnerFit<S>` so it composes
/// with `ShapeAwareEnvelopeScore` unchanged.
pub struct OcatInner<S: LinearSolver = CholeskySolver> {
    pub x_design: Array2<f64>,
    pub y: Array1<f64>,
    pub prior_weights: Option<Array1<f64>>,
    /// Per-term penalty blocks. Currently `s_list.len() == 1` is required —
    /// the ocat inner solver hasn't been lifted to true multi-smooth yet
    /// (would need a per-term gradient assembly inside the joint
    /// β + threshold step). Tracked as a follow-up to 94b.
    pub s_list: Vec<Array2<f64>>,
    /// Family aggregator — `loss.thresholds` is read on every iteration.
    /// `Family<OcatLoss, IdentityLink, OcatVariance>` is the only valid
    /// combination (no other link / variance impl makes sense).
    pub family: Family<OcatLoss, IdentityLink, OcatVariance>,
    pub opts: PirlsOpts,
    pub _solver: PhantomData<S>,
}

impl<S: LinearSolver> InnerSolver for OcatInner<S> {
    type Fit = GaussianInnerFit<S>;

    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit> {
        debug_assert_eq!(
            rho.len(),
            self.s_list.len(),
            "OcatInner: rho length {} must equal s_list length {}",
            rho.len(),
            self.s_list.len()
        );
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        self.ocat_loop(s_total)
    }
}

impl<S: LinearSolver> OcatInner<S> {
    fn ocat_loop(&self, s_total: Array2<f64>) -> Result<GaussianInnerFit<S>> {
        // λ is absorbed into s_total; the algebra below treats `λS` as
        // `1.0 · s_total` consistently.
        let lambda = 1.0_f64;
        let n = self.x_design.nrows();
        let p = self.x_design.ncols();
        let r = self.family.loss.n_cats;
        let theta = &self.family.loss.thresholds;
        debug_assert_eq!(theta.len(), r - 2);
        let alpha = self.family.loss.alpha();
        let prior_w: Array1<f64> = match &self.prior_weights {
            Some(w) => w.clone(),
            None => Array1::ones(n),
        };

        // Initial η — boundary midpoint per mgcv `efam.r:2947`. For interior
        // categories the midpoint is finite; the boundary categories use
        // `α_init_low = −2` and `α_init_high = α_R + 1` (mgcv's choice).
        let lo_inf = -2.0_f64;
        let hi_inf = alpha[r - 1] + 1.0;
        let mut eta = Array1::<f64>::zeros(n);
        for i in 0..n {
            let yi = (self.y[i].round() as i64).clamp(1, r as i64) as usize;
            let lo = if yi == 1 { lo_inf } else { alpha[yi - 1] };
            let hi = if yi == r { hi_inf } else { alpha[yi] };
            eta[i] = 0.5 * (lo + hi);
        }
        if let Some(e0) = &self.opts.eta_init {
            eta.assign(e0);
        }

        let mut beta = Array1::<f64>::zeros(p);
        let mut a_factor_opt: Option<S::Factorization> = None;
        let mut working_weights = Array1::<f64>::ones(n);
        let mut working_response = self.y.clone();
        let mut converged = false;
        let mut iters_used = 0;
        let mut dev_total = self.ocat_deviance(&eta, &prior_w);
        let mut pdev = dev_total + lambda * beta_sbeta(&s_total, &beta);

        for it in 0..self.opts.max_iters {
            // Build per-row Dmu / Dmu2 at the current η.
            let (dmu, dmu2) = self.ocat_dmu_dmu2(&eta);
            for i in 0..n {
                let w_i = (0.5 * dmu2[i] * prior_w[i]).max(1e-12);
                working_weights[i] = w_i;
                let denom = dmu2[i].max(1e-12);
                // Identity link, mgcv-style working response:
                //   z = η − Dmu / Dmu2  (NOT the standard `η + (y - μ)·g'(μ)`).
                working_response[i] = eta[i] - dmu[i] / denom;
            }

            // `(X' diag(w) X + λS + ridge·I) β = X' diag(w) z`.
            // v0.x-style adaptive ridge — `1e-5 · (1 + √n_pen) · max(|diag(X'WX)|)`
            // (mgcv_rust `pirls/setup.rs::pirls_ridge_scale`,
            // `build_penalised_a_with_ridge`). Regularises the linear
            // system when one λ_j saturates and X'WX + λS becomes
            // near-singular on the saturated subspace; without it the
            // solve drifts β toward an over-saturated optimum (closes
            // the 5.9% multi-smooth ocat parity gap, 2026-05-27).
            let (beta_trial, factor_trial) = {
                let xs = SplitDesign::new(self.x_design.view());
                let xtwx = xs.xtwx(working_weights.view());
                let xtwz = xs.t_dot_vec((&working_weights * &working_response).view());
                let n_pen = self.s_list.len() as f64;
                let ridge_scale = 1.0e-5 * (1.0 + n_pen.sqrt());
                let max_diag = xtwx.diag().iter().map(|x| x.abs()).fold(1.0_f64, f64::max);
                let ridge = ridge_scale * max_diag;
                let mut a = xtwx;
                add_penalty(&mut a, &s_total, lambda);
                for i in 0..p {
                    a[[i, i]] += ridge;
                }
                let factor = S::factorize(a)?;
                let b = S::solve(&factor, xtwz.view());
                (b, factor)
            };

            // Three-guard step-halving — generic `halve_until_valid` helper.
            // Ocat's validity is just "η finite for every obs" (μ = η for
            // identity link); no Loss-deviance probe like PIRLS, because the
            // ocat deviance is finite for any η.
            let pdev_old = pdev;
            let beta_old = beta.clone();
            let iter_one = it == 0;
            let beta_try0 = beta_trial.clone();
            let eta_try0 = self.x_design.dot(&beta_try0);
            let dev_try0 = self.ocat_deviance(&eta_try0, &prior_w);
            let pdev_try0 = dev_try0 + lambda * beta_sbeta(&s_total, &beta_try0);

            let recompute = |b: &Array1<f64>| {
                let e = self.x_design.dot(b);
                let d = self.ocat_deviance(&e, &prior_w);
                let pd = d + lambda * beta_sbeta(&s_total, b);
                (e, d, pd, None)
            };
            let is_invalid = |e: &Array1<f64>, _m: Option<&Array1<f64>>| -> bool {
                !e.iter().all(|ev| ev.is_finite())
            };

            let (beta_try, eta_try, dev_try, pdev_try, _mu, accepted) = halve_until_valid(
                beta_try0, &beta_old, eta_try0, dev_try0, pdev_try0, None, pdev_old, iter_one,
                recompute, is_invalid,
            );

            if accepted {
                let beta_max_change = beta_try
                    .iter()
                    .zip(beta_old.iter())
                    .map(|(b, bo)| (b - bo).abs())
                    .fold(0.0_f64, f64::max);
                beta = beta_try;
                eta = eta_try;
                a_factor_opt = Some(factor_trial);
                dev_total = dev_try;
                pdev = pdev_try;
                iters_used = it + 1;
                // mgcv-style convergence: max coef change < tol.
                if it > 0 && beta_max_change < self.opts.dev_rel_tol {
                    converged = true;
                    break;
                }
            } else {
                // Halving failed within `halve_until_valid`'s budget.
                // v0.x recipe (`pirls/mod.rs:2192-2197`): revert THIS step
                // to β_old, recompute η at the reverted β, and **continue**
                // the outer PIRLS loop. The next iter rebuilds (w, z) at
                // the reverted η — often producing a feasible step. Do
                // NOT break: breaking here aborted PIRLS at iter 1–2,
                // leaving β ≈ 0 and the term's smooth at edf ≈ 1 (this
                // was the 5.9% multi-smooth ocat parity gap, 2026-05-27).
                beta = beta_old;
                eta = self.x_design.dot(&beta);
                dev_total = self.ocat_deviance(&eta, &prior_w);
                pdev = dev_total + lambda * beta_sbeta(&s_total, &beta);
                iters_used = it + 1;
                // Don't update a_factor_opt — keep the previous successful
                // factor for vcov consumers.
            }
        }

        // Final-pass (w, z) at the converged β for the score body.
        let (dmu, dmu2) = self.ocat_dmu_dmu2(&eta);
        for i in 0..n {
            working_weights[i] = (0.5 * dmu2[i] * prior_w[i]).max(1e-12);
            let denom = dmu2[i].max(1e-12);
            working_response[i] = eta[i] - dmu[i] / denom;
        }
        // working_rss `Σ W (z − η)²` (mgcv `dev_num` analogue).
        let mut working_rss = 0.0_f64;
        for i in 0..n {
            let r = working_response[i] - eta[i];
            working_rss += working_weights[i] * r * r;
        }
        // μ = η for identity link.
        let mu = eta.clone();
        // FINAL-PASS factor: rebuild `A = X' diag(w_final) X + λS + ridge·I`
        // using `working_weights` AT THE CONVERGED η. v0.x does this in
        // `reml_criterion_ocat_proper`'s caller (`pirls/mod.rs:2231-2249`
        // → `evaluate_reml_ocat_proper_at` rebuilds A from w_final).
        // The factor stashed during the PIRLS loop was built from the
        // PREVIOUS iter's w (one η-step stale); using it for log|H| in
        // the score caused the +30 score drift on saturated-λ ocat
        // multi-smooth fits (parity report 2026-05-27).
        let a_factor = {
            // v0.x recipe: POST-penalty max_diag (`reml/ocat_joint.rs:284-303`).
            // The score's `log|H|` then matches v0.x byte-for-byte.
            // The analytic ρ-gradient must include the
            // `∂ridge/∂ρ_j · tr(H⁻¹) / 2` term because the post-penalty
            // ridge depends on λ — that term is wired in
            // `shape_aware.rs::compute_value_grad`.
            let mut a = SplitDesign::new(self.x_design.view()).xtwx(working_weights.view());
            add_penalty(&mut a, &s_total, lambda);
            let n_pen = self.s_list.len() as f64;
            let ridge_scale = 1.0e-5 * (1.0 + n_pen.sqrt());
            let max_diag_post_pen = a.diag().iter().map(|x| x.abs()).fold(1.0_f64, f64::max);
            let ridge = ridge_scale * max_diag_post_pen;
            for i in 0..p {
                a[[i, i]] += ridge;
            }
            S::factorize(a)?
        };
        // The during-loop factor is no longer used after the final-pass
        // refactor — drop it explicitly so the variable's purpose is clear.
        let _ = a_factor_opt;
        Ok(GaussianInnerFit::<S> {
            beta,
            eta,
            mu,
            // Factor and its weights travel together — see `a_weights`.
            a_weights: working_weights.clone(),
            // No separate expected curvature on this path — edf/vcov keep
            // using `a_factor`, as before. See `GaussianInnerFit::fisher`.
            fisher: None,
            working_weights,
            working_response,
            deviance: dev_total,
            rss: working_rss,
            n,
            p,
            iterations: iters_used,
            converged,
            a_factor,
            log_det_h_override: None,
            tk_kkt_inputs: None,
            // ocat goes through the shape-aware score path, not the
            // `EnvelopeScore` analytic Hessian — no `∂W/∂η` needed here.
            dw_deta: None,
            x_design: None,
            a_inv_cache: Default::default(),
        })
    }

    /// Sum of per-obs ocat deviance at the given η (μ = η for identity).
    fn ocat_deviance(&self, eta: &Array1<f64>, prior_w: &Array1<f64>) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..eta.len() {
            s += prior_w[i] * self.family.loss.deviance_per_obs(self.y[i], eta[i]);
        }
        s
    }

    /// Per-row `(Dmu, Dmu2)` at the current η. Reads thresholds from
    /// `self.family.loss`. Cancellation-safe via the same `abcd` helpers
    /// used by `Loss::d_loss_dmu`.
    fn ocat_dmu_dmu2(&self, eta: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let n = eta.len();
        let r = self.family.loss.n_cats;
        let alpha = self.family.loss.alpha();
        let mut dmu = Array1::<f64>::zeros(n);
        let mut dmu2 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let yi = (self.y[i].round() as i64).clamp(1, r as i64) as usize;
            let mu_i = eta[i];
            let al0 = alpha[yi - 1] - mu_i;
            let al1 = alpha[yi] - mu_i;
            let f = OcatLoss::fdiff_boundary(al0, al1).max(f64::MIN_POSITIVE);
            let a0 = OcatLoss::abcd_a(al0);
            let a1 = OcatLoss::abcd_a(al1);
            let b0 = OcatLoss::abcd_b(al0);
            let b1 = OcatLoss::abcd_b(al1);
            let a = a1 - a0;
            let b = b1 - b0;
            dmu[i] = -2.0 * a / f;
            dmu2[i] = 2.0 * (a * a / f - b) / f;
        }
        (dmu, dmu2)
    }
}
