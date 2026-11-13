//! Armijo-backtracking IRLS for ELF (Extended Log-F) quantile families.
//!
//! Ported from v0.x `fit_pirls_quantile` (`src/pirls/mod.rs:2397-2723`).
//!
//! Internally uses a safety-ridge Cholesky for the per-iter linear solve
//! (mgcv's `pirls_ridge_scale` shape — saturation can collapse ELF working
//! weights to ~0, and a tiny diagonal lift is the documented stabiliser).
//! The **emitted** `GaussianInnerFit<S>` uses the configured `S` backend
//! by re-factoring `A` once at convergence; this keeps the score body
//! backend-consistent without losing the in-loop safety ridge.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::{GamrsError, Result};
use crate::family::{ElfLoss, ElfVariance, Family, IdentityLink};
use crate::traits::InnerSolver;

use super::{
    add_penalty, chol_back_solve, chol_forward_solve, cholesky_with_safety_ridge, split::SplitDesign,
    CholeskySolver, GaussianInnerFit, LinearSolver,
};

/// `InnerSolver` for the qgam-style Quantile/ELF family.
///
/// At residual `r = y - η` (identity link), the ELF working IRLS
/// quantities derived analytically (Fasiolo et al. 2021):
///
/// ```text
///   s_i = sigmoid((y_i - η_i) / λ)
///   w_i = s_i(1 - s_i) / (σ λ)        (PSD Hessian; can go to 0 at saturation)
///   g_i = (s_i - (1 - τ)) / σ          ("working gradient", bounded)
///   Newton step solves: (X' W X + λ_pen S) β = X' (W·η + g)
/// ```
///
/// Reform of `X' W z` with `z = η + g/w` to avoid divide-by-zero when
/// `w → 0` at saturation. Per-iter Armijo backtracking on the penalised
/// ELF deviance — at extreme τ the full Newton step can overshoot into
/// the logistic's saturation regime and diverge on the next solve; the
/// halving prevents that.
///
/// Returns `GaussianInnerFit<S>` so it composes with `EnvelopeScore`.
pub struct ArmijoElfInner<S: LinearSolver = CholeskySolver> {
    pub x_design: Array2<f64>,
    pub y: Array1<f64>,
    pub prior_weights: Option<Array1<f64>>,
    /// Per-term penalty blocks. Multi-smooth quantile is supported: `fit`
    /// combines them as `Σ_j exp(ρ_j) S_j` via `combined_s` (one entry per
    /// smoothing parameter), and the Armijo objective penalises with that
    /// `s_total` — so the loop below is term-count agnostic.
    pub s_list: Vec<Array2<f64>>,
    /// Family carries (τ, σ, λ). `prior_weights` are NOT folded into the
    /// per-obs ELF weights — qgam's pinball loss doesn't have a natural
    /// "prior weight" interpretation; the field is kept for API consistency
    /// only and currently ignored.
    pub family: Family<ElfLoss, IdentityLink, ElfVariance>,
    pub opts: ArmijoElfOpts,
    /// Warm-start β. When supplied, skips the Gaussian-init linear solves
    /// inside `fit`. Used by the `fit_quantile_cr` driver to seed the
    /// τ-shifted Gaussian fit per v0.x's qgam-style warm start.
    pub beta_init: Option<Array1<f64>>,
    pub _solver: PhantomData<S>,
}

#[derive(Clone)]
pub struct ArmijoElfOpts {
    pub max_iters: usize,
    /// Max-β-change tolerance for convergence (mirrors v0.x).
    pub tol: f64,
    /// Max Armijo backtracking halvings per iteration. v0.x uses 20.
    pub max_halvings: usize,
}

impl Default for ArmijoElfOpts {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tol: 1e-7,
            max_halvings: 20,
        }
    }
}

impl<S: LinearSolver> InnerSolver for ArmijoElfInner<S> {
    type Fit = GaussianInnerFit<S>;

    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit> {
        debug_assert_eq!(
            rho.len(),
            self.s_list.len(),
            "ArmijoElfInner: rho length {} must equal s_list length {}",
            rho.len(),
            self.s_list.len()
        );
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        self.armijo_loop(s_total)
    }
}

impl<S: LinearSolver> ArmijoElfInner<S> {
    fn armijo_loop(&self, s_total: Array2<f64>) -> Result<GaussianInnerFit<S>> {
        // s_total absorbs `Σ_j λ_j S_j`; keep `lambda_pen = 1` to match
        // the single-smooth algebra below verbatim.
        let lambda_pen = 1.0_f64;
        let n = self.x_design.nrows();
        let p = self.x_design.ncols();
        let tau = self.family.loss.tau;
        let sigma = self.family.loss.sigma;
        let lambda_elf = self.family.loss.lambda;

        if !(tau > 0.0 && tau < 1.0) {
            return Err(GamrsError::InvalidParameter(format!(
                "ELF τ must be in (0, 1); got {tau}"
            )));
        }
        if sigma <= 0.0 || lambda_elf <= 0.0 {
            return Err(GamrsError::InvalidParameter(format!(
                "ELF σ and λ must be > 0; got σ={sigma} λ={lambda_elf}"
            )));
        }

        // β init: warm-start if provided, else zero. The fit_quantile_cr
        // driver always supplies a Gaussian-warm-start β; this branch
        // is only for direct trait-stack callers (smoke tests).
        let mut beta: Array1<f64> = self.beta_init.clone().unwrap_or_else(|| Array1::zeros(p));

        // Penalised ELF deviance — the Armijo objective.
        let elf_pen_deviance = |b: &Array1<f64>| -> f64 {
            let eta_t = self.x_design.dot(b);
            let mut total = 0.0_f64;
            for i in 0..n {
                let d =
                    crate::family::elf_parts(self.y[i], eta_t[i], tau, sigma, lambda_elf).deviance;
                if !d.is_finite() {
                    return f64::INFINITY;
                }
                total += d;
            }
            let sb = s_total.dot(b);
            let pen: f64 = b.iter().zip(sb.iter()).map(|(&bi, &sbi)| bi * sbi).sum();
            total + lambda_pen * pen
        };

        let mut obj_cur = elf_pen_deviance(&beta);
        let mut converged = false;
        let mut iter = 0usize;
        // Last accepted A (with safety ridge) — needed if we never re-build
        // A at convergence with the configured backend.
        let mut last_a: Option<Array2<f64>> = None;

        for outer_iter in 0..self.opts.max_iters {
            iter = outer_iter + 1;
            let eta: Array1<f64> = self.x_design.dot(&beta);

            // Per-obs working pieces. Saturation-safe (qgam elf.R:159-162):
            //   w_i goes to 0 at saturation; we use the working-gradient
            //   form X'(Wη + g) instead of X'Wz to dodge the divide.
            let mut w = Array1::<f64>::zeros(n);
            let mut g = Array1::<f64>::zeros(n);
            for i in 0..n {
                let s_i =
                    crate::family::elf_parts(self.y[i], eta[i], tau, sigma, lambda_elf).sigmoid;
                // w_i = s_i(1-s_i) / (σ λ)
                w[i] = s_i * (1.0 - s_i) / (sigma * lambda_elf);
                // g_i = (s_i - (1-τ)) / σ
                g[i] = (s_i - (1.0 - tau)) / sigma;
            }

            // X'WX + λ_pen S + ridge.
            let mut a = SplitDesign::new(self.x_design.view()).xtwx(w.view());
            add_penalty(&mut a, &s_total, lambda_pen);

            // RHS = X' (W·η + g) — well-defined at saturation.
            let weta_plus_g: Array1<f64> = w
                .iter()
                .zip(eta.iter())
                .zip(g.iter())
                .map(|((&wi, &etai), &gi)| wi * etai + gi)
                .collect();
            let xt_rhs = self.x_design.t().dot(&weta_plus_g);

            // Tiny ridge for stability when sat-weights collapse — same
            // shape as v0.x's `pirls_ridge_scale` / `build_penalised_a_with_ridge`.
            // The in-loop solve stays on Cholesky (with safety ridge) because
            // saturation makes A's smallest eigenvalues so small that Cholesky
            // is empirically the most reliable factor at this scale; the
            // configured `S` backend kicks in at convergence (one re-factor).
            let a_ridged = a.clone();
            let chol_trial = cholesky_with_safety_ridge(a_ridged, "ELF")?;
            let z = chol_forward_solve(&chol_trial, xt_rhs.view());
            let beta_proposed = chol_back_solve(&chol_trial, z.view());

            // Armijo backtracking on the penalised ELF objective. Direction
            // is the full Newton step; α starts at 1, halves until the
            // objective does not increase (per v0.x's `1e-10` slack).
            let direction: Vec<f64> = beta_proposed
                .iter()
                .zip(beta.iter())
                .map(|(&bn, &bo)| bn - bo)
                .collect();
            let mut alpha = 1.0_f64;
            let mut accepted = false;
            let mut beta_new = beta.clone();
            let mut obj_new = obj_cur;
            for _ in 0..self.opts.max_halvings {
                for j in 0..p {
                    beta_new[j] = beta[j] + alpha * direction[j];
                }
                obj_new = elf_pen_deviance(&beta_new);
                if obj_new.is_finite() && obj_new <= obj_cur + 1.0e-10 {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                // No descent direction found — declare convergence by
                // stagnation (matches v0.x).
                converged = true;
                break;
            }

            let max_change = beta_new
                .iter()
                .zip(beta.iter())
                .map(|(b, b_old)| (b - b_old).abs())
                .fold(0.0_f64, f64::max);
            beta = beta_new;
            obj_cur = obj_new;
            last_a = Some(a);

            if max_change < self.opts.tol {
                converged = true;
                break;
            }
        }

        // Final quantities. ELF identity-link: μ = η; the fitted "deviance"
        // is the ELF sum (gives a sensible score-formula input even though
        // there's no Gaussian-style RSS here).
        let eta: Array1<f64> = self.x_design.dot(&beta);
        let mu = eta.clone();
        let mut working_weights = Array1::<f64>::zeros(n);
        let mut working_response = Array1::<f64>::zeros(n);
        let mut deviance = 0.0_f64;
        for i in 0..n {
            let parts = crate::family::elf_parts(self.y[i], eta[i], tau, sigma, lambda_elf);
            // ELF working weight w_i = ∂²L/∂μ² / 2 (since deviance = 2L,
            // the working weight should be d²(L)/dμ² to match the
            // Fisher-info Hessian shape used in PIRLS).
            let w_i = 0.5 * parts.d2l_dmu;
            working_weights[i] = w_i;
            // working response: η + g/w  where  g = (s - (1-τ))/σ, well-defined
            // when w > 0.
            let g_i = (parts.sigmoid - (1.0 - tau)) / sigma;
            working_response[i] = if w_i > 1e-12 {
                eta[i] + g_i / w_i
            } else {
                eta[i]
            };
            deviance += parts.deviance;
        }

        // Re-factor A with the configured `S` backend at the converged β so
        // the score body sees a backend-consistent factor (the in-loop
        // safety-ridge Cholesky was for stability, not the surface the
        // score reads). Match v0.x's pattern: tiny diag ridge for stability
        // here too, then `S::factorize`.
        let a_for_factor = match last_a {
            Some(a) => a,
            None => {
                // Never-accepted path: rebuild A at converged β.
                let mut a = SplitDesign::new(self.x_design.view()).xtwx(working_weights.view());
                add_penalty(&mut a, &s_total, lambda_pen);
                a
            }
        };
        let mut a_final = a_for_factor;
        let max_diag = a_final
            .diag()
            .iter()
            .map(|v| v.abs())
            .fold(1.0_f64, f64::max);
        let ridge = 1e-7 * max_diag;
        for i in 0..p {
            a_final[[i, i]] += ridge;
        }
        let a_factor = S::factorize(a_final)?;

        let _ = self.prior_weights.as_ref();

        // Per-obs `∂W/∂η` for the analytic outer-Newton Hessian's W-chain.
        // ELF is identity-link (μ = η) with `W = ½·∂²L/∂μ²`, so
        // `∂W/∂η = ½·∂³L/∂μ³`. We get the third derivative by a tight
        // central difference of `½·d2l_dmu` (the ELF sigmoid pieces are
        // smooth; analytic d3 is not exposed by `elf_parts`).
        let mut dw_deta = Array1::<f64>::zeros(n);
        let hfd = 1e-5_f64;
        for i in 0..n {
            let wp = 0.5
                * crate::family::elf_parts(self.y[i], eta[i] + hfd, tau, sigma, lambda_elf).d2l_dmu;
            let wm = 0.5
                * crate::family::elf_parts(self.y[i], eta[i] - hfd, tau, sigma, lambda_elf).d2l_dmu;
            dw_deta[i] = (wp - wm) / (2.0 * hfd);
        }

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
            deviance,
            rss: deviance, // ELF has no RSS; reuse the deviance.
            n,
            p,
            iterations: iter,
            converged,
            a_factor,
            log_det_h_override: None,
            tk_kkt_inputs: None,
            dw_deta: Some(dw_deta),
            x_design: Some(self.x_design.clone()),
        })
    }
}
