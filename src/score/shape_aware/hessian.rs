//! Hessian paths for `ShapeAwareEnvelopeScore`.
//!
//! Entry point is `compute_value_grad_hess_analytical` — one PIRLS solve
//! at θ produces (value, gradient, FrozenBetaCtx); the Hessian then
//! comes from a central FD variant chosen by whether the family
//! supplies `analytic_shape_score_gradient`.
//!
//! `compute_value_grad_hess_rho_only` is the **profile-θ** variant: it
//! returns just the M×M ρ block of the joint Hessian, the ρ-only gradient,
//! and the value — no shape FD probes. Used by the `ProfileShapeNewton`
//! outer solver (port of mgcv_rust `src/smooth.rs:1866-1869`'s "legacy
//! M-dim Newton path" comment, plus the NegBin 1-D profile-θ Newton at
//! lines 3562-3637).

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::inner::{GaussianInnerFit, LinearSolver};
use crate::traits::{shape_pair_index, Link, Loss, VarianceFn};

use super::super::hess_ift::{
    build_xtwx, compute_dev_grad_beta_working_rss, hess_ift_rho, HessIftCtx,
};
use super::super::profile::Profile;
use super::builder::ShapeInnerBuilder;
use super::score::{FrozenBetaCtx, ShapeAwareEnvelopeScore};

impl<L, K, V, B, P, S> ShapeAwareEnvelopeScore<L, K, V, B, P, S>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Coupled `(value, grad, hess)` — three-way dispatch chosen at the
    /// per-family level to match mgcv_rust's PIRLS-economy pattern.
    ///
    /// Recipe (per v0.x `src/reml/tweedie_joint.rs::
    /// tweedie_theta_grad_hess_analytic` and mgcv_rust's NegBin path at
    /// `src/smooth.rs:1866-1869` / `3562-3639`): ONE PIRLS solve at
    /// θ_center → value + gradient. Hessian dispatch then matches the
    /// cheapest correct FD pattern given what the family supplies:
    ///
    /// 1. **Closed-form shape-grad** (`Loss::analytic_shape_score_gradient
    ///    = Some`): Tweedie. Hessian via partial-freeze FD on the
    ///    analytic gradient — log-λ row re-converges PIRLS (2·M solves);
    ///    shape rows freeze β̂ (0 solves). Total: 1 + 2·M PIRLS.
    /// 2. **Level-1 IFT shape-grad** (`level1_shape_derivatives = Some`):
    ///    NegBin, scat/TDist, Ocat. The shape gradient is analytic via
    ///    the IFT path (`analytic_shape_grad_via_ift` at gradient.rs:278),
    ///    so FD on `compute_value_grad` is FD-on-analytic — no chained-FD
    ///    noise concern that v0.1's `hess_via_fd_on_value` was guarding
    ///    against (it predates the IFT analytic shape gradient, added in
    ///    commit 85946a1). Total: 1 + 2·d PIRLS solves (vs `on_value`'s
    ///    `1 + 2·d²`). For NegBin d=2 (1-D smooth + 1 shape): 5 vs 9
    ///    PIRLS. For d=3 (2-D smooth + 1 shape): 7 vs 19 — matches the
    ///    "M-dim Newton + cheap shape-side" PIRLS economy mgcv_rust gets
    ///    in `src/smooth.rs:2383` (analytic ρ-Hessian via
    ///    `reml_hessian_mgcv_exact_ift`) plus `3562-3639` (3-evals 1-D
    ///    profile-θ Newton).
    /// 3. **No analytic gradient anywhere**: fall back to direct FD on
    ///    the REML score value. None of gamrs's shipped families hit
    ///    this branch today — kept as a safety net for hypothetical
    ///    Loss impls without `level1_shape_derivatives`.
    ///
    /// `Loss::analytic_shape_score_hessian` is an optional override for
    /// the shape×shape block (defaults `None`; currently no gamrs family
    /// uses it — v0.x's FD-on-analytic-grad converges without it).
    pub(super) fn compute_value_grad_hess_analytical(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let _phase_total = crate::profile::scoped("value_grad_hess_total");
        let (fit, family) = {
            let _t = crate::profile::scoped("fit_inner_at");
            self.fit_inner_at(theta)?
        };
        let n_terms = self.s_list.len();
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let value = {
            let _t = crate::profile::scoped("score_value");
            self.score_value(&fit, &family, &rho_slice)
        };

        // **Tier 1a cache**: compute Level-1 derivatives ONCE here and
        // share between the gradient assembly (`eval_grad_with_fit_cached`)
        // and the Hessian dispatch (both consumers need it at the same
        // θ; this eliminates the second `family.loss.level1_shape_derivatives`
        // call that was costing ~50-100 μs per outer iter on scat).
        let n_shape = family.n_shape_params();
        let level1_cached = if n_shape > 0 {
            let _t = crate::profile::scoped("level1_compute_once");
            family.loss.level1_shape_derivatives(
                self.y.view(),
                fit.eta.view(),
                self.prior_weights.as_ref().map(|w| w.view()),
            )
        } else {
            None
        };

        let (g_center, ctx) = {
            let _t = crate::profile::scoped("eval_grad_with_fit");
            self.eval_grad_with_fit_cached(theta, &fit, &family, level1_cached.as_ref())?
        };

        let has_analytic_shape_grad = n_shape == 0
            || family
                .loss
                .analytic_shape_score_gradient(
                    self.y.view(),
                    fit.mu.view(),
                    fit.deviance,
                    1.0,
                    ctx.phi_center,
                )
                .is_some();
        // The IFT analytic shape-gradient path fires when the family
        // supplies `level1_shape_derivatives` — already cached above.
        let has_ift_shape_grad = n_shape == 0 || level1_cached.is_some();

        // Per-family opt-in: families with `prefers_full_fd_hessian = true`
        // skip the analytic / partial-FD paths and use full FD on the REML
        // score value. Ocat takes this route (matching mgcv_rust's
        // `reml_joint_ocat_finite_diff`) because its ordered-threshold
        // surface has a near-flat coordinated-shift ridge that the IFT
        // path's sparse off-diagonal Hessian doesn't stabilise.
        let _t_hess = crate::profile::scoped("hess_dispatch");
        let mut hess = if family.loss.prefers_full_fd_hessian() {
            self.hess_via_fd_on_value(theta)?
        } else if has_analytic_shape_grad {
            // Tweedie path: 2·M PIRLS + 0 shape solves.
            self.hess_via_fd_frozen_beta(theta, &fit, &ctx)?
        } else if has_ift_shape_grad {
            // NegBin / scat / Ocat path. Two sub-paths:
            //
            //   (i)  **Full Level-2 analytic** when the family supplies
            //        `level2_shape_derivatives` AND `level1_shape_derivatives`
            //        — closed-form joint (M+n_shape)×(M+n_shape) Hessian
            //        via mgcv R's `gdi2` chain rule (port of mgcv_rust
            //        `tdist_gdi2_native`). Zero PIRLS, zero FD on shape
            //        axes. Requires the family's PIRLS to use the same A
            //        the Level-1 / Level-2 derivatives were derived under
            //        (`W = ½·D_μμ`) — TDist routes through
            //        `irls_observed_pair` to deliver that.
            //
            //   (ii) **Analytic-ρ + frozen-β IFT shape FD** when only
            //        Level-1 is supplied (NegBin / Ocat today). Same cost
            //        profile as (i) but the shape rows come from central
            //        FD of the analytic IFT gradient at frozen β̂
            //        (introduced in v0.11).
            // Level-1 is the cached one from above (Tier 1a). Level-2
            // is family-only and computed fresh here — it's only needed
            // for the Hessian path and isn't shared.
            let level2 = family.loss.level2_shape_derivatives(
                self.y.view(),
                fit.eta.view(),
                self.prior_weights.as_ref().map(|w| w.view()),
            );
            match (level1_cached.as_ref(), level2) {
                (Some(lv1), Some(lv2)) => {
                    self.hess_via_ift_level2(theta, &fit, &family, lv1, &lv2, n_shape)?
                }
                _ => self.hess_via_ift_analytic(theta, &fit, &family, n_shape, &ctx)?,
            }
        } else {
            // Safety-net path: direct FD on REML value.
            self.hess_via_fd_on_value(theta)?
        };

        // Optional family-supplied closed-form shape×shape block. Lives
        // at hess[n_terms..n_terms+n_shape, n_terms..n_terms+n_shape].
        // Currently `None` for every gamrs family — hook for future ports.
        if n_shape > 0 {
            let dp = fit.deviance + ctx.bsb_total;
            if let Some(block) = family.loss.analytic_shape_score_hessian(
                self.y.view(),
                fit.mu.view(),
                dp,
                ctx.n_minus_mp,
                ctx.phi_center,
            ) {
                debug_assert_eq!(block.shape(), &[n_shape, n_shape]);
                for j in 0..n_shape {
                    for k in 0..n_shape {
                        hess[[n_terms + j, n_terms + k]] = block[[j, k]];
                    }
                }
            }
        }

        Ok((value, g_center, hess))
    }

    /// ρ-only `(value, g_ρ, H_ρρ)` at θ for the profile-shape Newton path.
    ///
    /// Port of mgcv_rust's NegBin profile-θ pattern: outer Newton steps
    /// **only ρ**, then a separate 1-D log(θ) Newton runs sequentially.
    /// Citations:
    /// - `src/smooth.rs:2383` — `reml_hessian_mgcv_exact_ift` returns an
    ///   M×M ρ-Hessian for NegBin (no log θ axis).
    /// - `src/smooth.rs:1866-1869` — comment block confirms NegBin /
    ///   Tweedie profile blocks run their 1-D log-θ Newton AFTER the ρ
    ///   step ("`joint_active = false` falls through to the legacy M-dim
    ///   Newton path").
    /// - `src/smooth.rs:3562-3637` — the actual NegBin profile-θ Newton
    ///   block (3 PIRLS for central FD on `dlr/d(log θ)`).
    ///
    /// PIRLS economy: **1 PIRLS solve total**. The full `value_grad_hess`
    /// path costs 1 + 2 + 2·n_shape = 5 PIRLS for NegBin 1-D (n_shape=1,
    /// n_terms=1) and 1 + 2 + 2·n_shape = 5 for NegBin 2-D (n_shape=1,
    /// n_terms=2) — the two extras come from `eval_grad_with_fit`'s
    /// shape-FD fallback (gradient.rs:564-574) and `hess_via_ift_analytic`'s
    /// shape-FD column (hessian.rs:474-487). Both vanish in the ρ-only
    /// path because the shape gradient/Hessian are no longer needed; the
    /// 1-D profile-θ Newton evaluates the REML *value* at log θ ± h
    /// (3 PIRLS) outside this function, matching mgcv_rust:3592-3594's
    /// `nb_eval!(log_theta), nb_eval!(log_theta + h), nb_eval!(log_theta - h)`.
    pub fn compute_value_grad_hess_rho_only(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        self.compute_value_grad_hess_rho_only_with_fit(theta)
            .map(|(v, g, h, _)| (v, g, h))
    }

    /// Same as `compute_value_grad_hess_rho_only` but also returns the
    /// converged inner fit.
    pub fn compute_value_grad_hess_rho_only_with_fit(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>, GaussianInnerFit<S>)> {
        use super::super::hess_ift::{
            build_xtwx, compute_dev_grad_beta_working_rss, hess_ift_rho, HessIftCtx,
        };

        let _t_total = crate::profile::scoped("rho_only_total");
        let (fit, family) = self.fit_inner_at(theta)?;
        let n_terms = self.s_list.len();
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let value = self.score_value(&fit, &family, &rho_slice);

        // ρ-gradient via the shared envelope helper. Per-term `bsb_j`
        // and `tr(H⁻¹·S_j)` are computed here; φ comes from the active
        // Profile (FixedAtOneProfile for NegBin → φ=1).
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut tr_hinv_s_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut bsb_total = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = self.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let tr_hinv_s_j = fit.trace_a_inv(self.s_list[j].view());
            bsb_per_term.push(bsb_j);
            tr_hinv_s_per_term.push(tr_hinv_s_j);
            bsb_total += rho_slice[j].exp() * bsb_j;
        }
        let tr_hinv_xtwx = fit.p as f64;
        let phi = self
            .profile
            .dispersion(&family.loss, &fit, 1.0, bsb_total, tr_hinv_xtwx, self.mp)
            .unwrap_or(1.0);
        let rho_grad = self.compute_rho_envelope_gradient(
            &fit,
            &family,
            &rho_slice,
            &bsb_per_term,
            &tr_hinv_s_per_term,
            phi,
        );
        let mut g_rho = Array1::<f64>::zeros(n_terms);
        for j in 0..n_terms {
            g_rho[j] = rho_grad[j];
        }

        // Analytic M×M ρ-Hessian via the IFT helper — identical to the
        // ρ block built by `hess_via_ift_analytic` (just no shape FD
        // column). Newton-A path for `use_newton_irls()` families (NegBin
        // here): the IFT Hessian differentiates the same `log|H|` the
        // score formula uses (Newton-W, not Fisher-W).
        let lambda: Vec<f64> = rho_slice.iter().map(|&r| r.exp()).collect();
        let sigma2 = phi; // ScaleParameterMethod::Profile uses φ from above.
        let use_newton = family.loss.use_newton_irls();
        let lazy_tk = if use_newton {
            let prior_w = self
                .prior_weights
                .clone()
                .unwrap_or_else(|| Array1::ones(fit.n));
            let rho_arr = Array1::from(rho_slice.clone());
            let s_total = crate::design::combined_s(&self.s_list, &rho_arr, self.x_design.ncols());
            crate::inner::pirls::lazy_tk_kkt_inputs(
                &family,
                &self.y,
                &fit.mu,
                &fit.beta,
                &prior_w,
                &self.x_design,
                &self.s_list,
                &s_total,
                &rho_arr,
            )
        } else {
            None
        };
        let (a_inv_owned, xtwx_owned, dev_grad_beta) = if let Some(ref tk) = lazy_tk {
            let a_inv = tk.a_newton_inv.clone();
            let n = fit.n;
            let mut w_newton = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mu_i = fit.mu[i];
                let var_i = family.variance.variance(mu_i).max(1e-300);
                let g_prime_mu = family.link.d_link_dmu(mu_i);
                let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
                let v_prime = family.variance.d_variance(mu_i);
                let v1n = v_prime / var_i;
                let g_double_prime = family.link.d2_link_dmu(mu_i);
                let g2n = g_double_prime / g_prime_mu;
                let c_resid = self.y[i] - mu_i;
                let alpha_raw = 1.0 + c_resid * (v1n + g2n);
                let alpha = if alpha_raw > 0.0 && alpha_raw.is_finite() {
                    alpha_raw
                } else {
                    1.0
                };
                w_newton[i] = wf * alpha;
            }
            let xtwx_n = build_xtwx(&self.x_design, &w_newton);
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &w_newton,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_n, dev_grad)
        } else {
            let a_inv = fit.a_inv();
            let xtwx_f = build_xtwx(&self.x_design, &fit.working_weights);
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &fit.working_weights,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_f, dev_grad)
        };
        let ctx = HessIftCtx {
            s_list: &self.s_list,
            lambda: &lambda,
            beta: &fit.beta,
            a_inv: &a_inv_owned,
            xtwx: &xtwx_owned,
            sigma2,
            dev_grad_beta: &dev_grad_beta,
        };
        let h_rho = {
            let _t = crate::profile::scoped("hess_ift_rho");
            hess_ift_rho(&ctx)
        };

        // Capture NoRefresh accepted state — port of mgcv_rust's
        // `warm_state` write (`gam_optimized.rs:1408-1414`). b1[:, k] =
        // -λ_k · A⁻¹ · S_k · β is the same IFT first-derivative used
        // inside `hess_ift_rho` above; we recompute here from the already-
        // materialised `a_inv_owned` (cheap: m × O(p²) GEMVs).
        //
        // Stored as Newton-A b1 when `use_newton_irls()` (matches the A
        // used in the Hessian and the score's log|H| override); Fisher-A
        // b1 otherwise. NoRefresh consumers re-validate η/μ before
        // accepting the propagated β, so a mild A mismatch from sigmoidal
        // β regions degrades gracefully via the guardrail.
        if family.loss.allows_no_refresh() {
            let p = fit.beta.len();
            let mut b1 = Array2::<f64>::zeros((p, n_terms));
            for k in 0..n_terms {
                let s_k_beta: Array1<f64> = self.s_list[k].dot(&fit.beta);
                let ainv_sk_beta: Array1<f64> = a_inv_owned.dot(&s_k_beta);
                let lam_k = lambda[k];
                for r in 0..p {
                    b1[[r, k]] = -lam_k * ainv_sk_beta[r];
                }
            }
            let shape_slice: Vec<f64> = theta.iter().skip(n_terms).copied().collect();
            *self.accepted_state.borrow_mut() = Some(super::score::AcceptedState {
                beta: fit.beta.clone(),
                b1,
                lambda: lambda.clone(),
                shape_params: shape_slice,
            });
        }

        Ok((value, g_rho, h_rho, fit))
    }

    /// **Partial-freeze** central FD on the analytic gradient.
    ///
    /// - Log-λ row/column (`i == 0`): re-converge PIRLS at θ ± h — the
    ///   β-chain through `dβ̂/dλ` matters far from the optimum (the
    ///   penalty acts directly on β), and freezing β̂ here would make
    ///   Newton stall (verified by canonical_api::tweedie failing in
    ///   the 2026-05-25 v0.x port). Cost: 2 PIRLS solves for this row.
    ///
    /// - Shape rows/columns (`i ≥ 1`): freeze β̂ at θ_center, evaluate
    ///   the analytic shape-gradient at perturbed shape params. The
    ///   envelope theorem makes this exact in the gradient and O(h) in
    ///   the Hessian — mirrors v0.x `tweedie_theta_grad_hess_analytic`
    ///   (`src/reml/tweedie_joint.rs:347-486`). Cost: 0 PIRLS solves
    ///   for these 2·(d-1) entries; only Wright-series / closed-form
    ///   evaluations per probe.
    ///
    /// Total: **2 PIRLS solves per outer Newton iter** (vs v0.1's 2d
    /// + 1 = 7 for Tweedie d=3). The dropped 4 PIRLS solves per outer
    /// iter are the speedup. The off-diagonal log-λ↔shape Hessian
    /// entries fill in symmetrically from the log-λ row (which is
    /// computed with re-converge, so it's correct).
    fn hess_via_fd_frozen_beta(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        ctx: &FrozenBetaCtx,
    ) -> Result<Array2<f64>> {
        let d = theta.len();
        let n_terms = self.s_list.len();
        let mut hess = Array2::<f64>::zeros((d, d));
        // Re-converge for each log-λ direction (i ∈ 0..n_terms).
        let eps_rho = 1.0e-4;
        for i in 0..n_terms {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps_rho;
            t_minus[i] -= eps_rho;
            let (_, g_plus_rho) = self.compute_value_grad(&t_plus)?;
            let (_, g_minus_rho) = self.compute_value_grad(&t_minus)?;
            for j in 0..d {
                hess[[j, i]] = (g_plus_rho[j] - g_minus_rho[j]) / (2.0 * eps_rho);
            }
        }
        // Frozen-β̂ for shape directions (i ∈ n_terms..d).
        let eps_shape = 1.0e-5;
        for i in n_terms..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps_shape;
            t_minus[i] -= eps_shape;
            let g_plus = self.eval_grad_frozen_beta(&t_plus, fit, ctx)?;
            let g_minus = self.eval_grad_frozen_beta(&t_minus, fit, ctx)?;
            for j in 0..d {
                hess[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * eps_shape);
            }
        }
        // Symmetrise — off-diagonal log-λ↔shape gets a clean average.
        for i in 0..d {
            for j in i + 1..d {
                let avg = 0.5 * (hess[[i, j]] + hess[[j, i]]);
                hess[[i, j]] = avg;
                hess[[j, i]] = avg;
            }
        }
        Ok(hess)
    }

    /// Direct central FD of the REML score value (no chained FD through
    /// the gradient). Mirrors v0.x's `reml_joint_ocat_finite_diff`
    /// (`src/smooth.rs:622-694`) for families without
    /// `analytic_shape_score_gradient` (scat / negbin / ocat).
    ///
    /// Diagonal: `(s(θ+h·eᵢ) − 2·s(θ) + s(θ−h·eᵢ)) / h²`.
    /// Off-diagonal: `(s(θ+h·eᵢ+h·eⱼ) − s(θ+h·eᵢ−h·eⱼ)
    ///               − s(θ−h·eᵢ+h·eⱼ) + s(θ−h·eᵢ−h·eⱼ)) / (4 h²)`.
    ///
    /// Cost: `1 + 2d + 2·d(d-1)` score evaluations (each a full PIRLS).
    /// For d=4 (ocat with 2 smooths) that's 33 PIRLS solves — heavy,
    /// but matches v0.x exactly and removes the chained-FD noise that
    /// made gamrs's outer Newton drift on the saturated-λ axis (parity
    /// report 2026-05-27).
    fn hess_via_fd_on_value(&self, theta: &Array1<f64>) -> Result<Array2<f64>> {
        let d = theta.len();
        let mut h = Array2::<f64>::zeros((d, d));
        let eps = 1.0e-4;
        let s0 = self.compute_value(theta)?;
        // Cache one-axis perturbations — reused for both the diagonal and
        // each off-diagonal mixed-difference.
        let mut s_plus = vec![0.0_f64; d];
        let mut s_minus = vec![0.0_f64; d];
        for i in 0..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps;
            t_minus[i] -= eps;
            s_plus[i] = self.compute_value(&t_plus)?;
            s_minus[i] = self.compute_value(&t_minus)?;
            h[[i, i]] = (s_plus[i] - 2.0 * s0 + s_minus[i]) / (eps * eps);
        }
        // Off-diagonal mixed central differences.
        for i in 0..d {
            for j in i + 1..d {
                let mut t_pp = theta.clone();
                let mut t_pm = theta.clone();
                let mut t_mp = theta.clone();
                let mut t_mm = theta.clone();
                t_pp[i] += eps;
                t_pp[j] += eps;
                t_pm[i] += eps;
                t_pm[j] -= eps;
                t_mp[i] -= eps;
                t_mp[j] += eps;
                t_mm[i] -= eps;
                t_mm[j] -= eps;
                let s_pp = self.compute_value(&t_pp)?;
                let s_pm = self.compute_value(&t_pm)?;
                let s_mp = self.compute_value(&t_mp)?;
                let s_mm = self.compute_value(&t_mm)?;
                let off = (s_pp - s_pm - s_mp + s_mm) / (4.0 * eps * eps);
                h[[i, j]] = off;
                h[[j, i]] = off;
            }
        }
        Ok(h)
    }

    /// Central FD on the gradient with FULL PIRLS re-converge at each
    /// ±h probe. Active for families whose `compute_value_grad` returns
    /// an analytic gradient end-to-end (NegBin / scat / Ocat via the IFT
    /// path at `gradient.rs:278`). Cost: `2·d` PIRLS solves per outer
    /// Newton iter (vs `hess_via_fd_on_value`'s `1 + 2·d²`). Tweedie
    /// uses the cheaper `hess_via_fd_frozen_beta` instead (closed-form
    /// shape-grad bypasses the shape-row PIRLS solves entirely).
    ///
    /// Kept as a fallback / reference for `hess_via_ift_analytic` parity.
    #[allow(dead_code)]
    fn hess_via_fd_on_grad(&self, theta: &Array1<f64>) -> Result<Array2<f64>> {
        let d = theta.len();
        let mut h = Array2::<f64>::zeros((d, d));
        let eps = 1.0e-4;
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

    /// Hybrid Hessian: analytic IFT for the M×M ρ block + FD-on-grad for
    /// shape rows/cols.
    ///
    /// Mathematical decomposition of the joint (ρ, θ_shape) Hessian:
    ///
    /// ```text
    ///   H = [ H_ρρ   H_ρθ ]    H_ρρ : M × M     ← analytic IFT
    ///       [ H_θρ   H_θθ ]    H_θρ = H_ρθ'     ← FD-on-grad probes
    ///                          H_θθ : n_shape × n_shape
    /// ```
    ///
    /// - **H_ρρ** comes from [`super::super::hess_ift::hess_ift_rho`], a
    ///   line-by-line port of mgcv_rust `reml_hessian_mgcv_exact_ift`
    ///   (`src/reml/mod.rs:2511-2813`). Zero PIRLS solves.
    /// - **H_θθ and H_ρθ** come from `2·n_shape` central-FD probes on the
    ///   analytic gradient along the shape axes only. Each probe runs
    ///   PIRLS to convergence at perturbed θ_shape, which captures the
    ///   full β-chain through `dβ/dθ_shape` (envelope theorem keeps the
    ///   shape ρ-gradient exact to O(h) when the shape gradient is
    ///   analytic via `analytic_shape_grad_via_ift`).
    /// - **H_θρ** is filled from the FD-on-grad probes' ρ entries of
    ///   `g_plus / g_minus`, then symmetrised with H_ρθ for numerical
    ///   stability (matches `hess_via_fd_on_grad`'s symmetrise pass).
    ///
    /// Total cost: `2 · n_shape` PIRLS solves per outer-Newton iter, vs
    /// `hess_via_fd_on_grad`'s `2 · (M + n_shape)`. NegBin 1-D (M=1,
    /// n_shape=1) drops from 4 to 2 PIRLS; NegBin 2-D from 6 to 2.
    ///
    /// **A consistency choice**: when the loss is on the Newton-IRLS path
    /// (NegBin / TDist), the IFT helper uses the **Newton A** (NOT Fisher
    /// A from `fit.a_factor`) so it differentiates the SAME `log|H|`
    /// the score formula uses (`score.rs:265-282`). The Newton A is
    /// materialised via `lazy_tk_kkt_inputs` (gradient.rs:314), matching
    /// the analytic shape-gradient path's choice. For Ocat (Fisher==Newton)
    /// `fit.a_factor` is used directly.
    fn hess_via_ift_analytic(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        family: &crate::family::Family<L, K, V>,
        n_shape: usize,
        frozen_ctx: &FrozenBetaCtx,
    ) -> Result<Array2<f64>> {
        let d = theta.len();
        let n_terms = self.s_list.len();
        debug_assert_eq!(d, n_terms + n_shape);
        let mut h = Array2::<f64>::zeros((d, d));

        // -----------------------------------------------------------------
        // 1) Analytic M×M ρ block via the IFT Hessian helper.
        // -----------------------------------------------------------------
        // Build the score's `λ_j = exp(ρ_j)` and σ². Mirrors the gradient
        // path's `bsb_total` aggregation (gradient.rs:200) and the score's
        // dispersion read (`Profile::dispersion`).
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let lambda: Vec<f64> = rho_slice.iter().map(|&r| r.exp()).collect();
        let mut bsb_total = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = self.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            bsb_total += lambda[j] * bsb_j;
        }
        let sigma2 = self
            .profile
            .dispersion(&family.loss, fit, 1.0, bsb_total, fit.p as f64, self.mp)
            .unwrap_or(1.0);

        // Newton-A path for `use_newton_irls()` families (NegBin, TDist).
        // Falls back to Fisher A for Ocat. Mirrors `analytic_shape_grad_via_ift`
        // (gradient.rs:298-329): the score's `log|H|` is the Newton one for
        // these families, so the IFT Hessian MUST differentiate the SAME A.
        let use_newton = family.loss.use_newton_irls();
        let lazy_tk = if use_newton {
            let prior_w = self
                .prior_weights
                .clone()
                .unwrap_or_else(|| Array1::ones(fit.n));
            let rho_arr = Array1::from(rho_slice.clone());
            let s_total = crate::design::combined_s(&self.s_list, &rho_arr, self.x_design.ncols());
            crate::inner::pirls::lazy_tk_kkt_inputs(
                family,
                &self.y,
                &fit.mu,
                &fit.beta,
                &prior_w,
                &self.x_design,
                &self.s_list,
                &s_total,
                &rho_arr,
            )
        } else {
            None
        };

        // Materialise `xtwx` and `a_inv` consistently. Newton path: pull
        // `A⁻¹` from `lazy_tk` (which already built A_newton⁻¹), rebuild
        // `w_newton` to derive `xtwx_newton = X' diag(w_newton) X`. Fisher
        // path: build xtwx from `fit.working_weights` and use `fit.a_inv()`.
        // `dev_grad_beta` then uses the SAME W/working_response pair so
        // the term-2 `(dev_grad)·b2` piece stays consistent.
        let (a_inv_owned, xtwx_owned, dev_grad_beta) = if let Some(ref tk) = lazy_tk {
            let a_inv = tk.a_newton_inv.clone();
            // Newton working weights — port of `lazy_tk_kkt_inputs`'s internal
            // `w_newton` (pirls.rs:171-187). Not exposed on TkKKTInputs so we
            // re-derive here using the same `wf · α` formula with the Fisher
            // fallback when α ≤ 0 (matching pirls.rs:268-271 alpha clamp).
            let n = fit.n;
            let mut w_newton = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mu_i = fit.mu[i];
                let var_i = family.variance.variance(mu_i).max(1e-300);
                let g_prime_mu = family.link.d_link_dmu(mu_i);
                let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
                let v_prime = family.variance.d_variance(mu_i);
                let v1n = v_prime / var_i;
                let g_double_prime = family.link.d2_link_dmu(mu_i);
                let g2n = g_double_prime / g_prime_mu;
                let c_resid = self.y[i] - mu_i;
                let alpha_raw = 1.0 + c_resid * (v1n + g2n);
                let alpha = if alpha_raw > 0.0 && alpha_raw.is_finite() {
                    alpha_raw
                } else {
                    1.0
                };
                w_newton[i] = wf * alpha;
            }
            let xtwx_n = build_xtwx(&self.x_design, &w_newton);
            // Newton normal equation: `X'·W_n·(z_n − Xβ) = λSβ` at converged β.
            // `fit.working_response` was stamped at the Newton fixed point
            // (pirls.rs:567 / :707), so `compute_dev_grad_beta_working_rss`
            // returns `-2·λSβ` here.
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &w_newton,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_n, dev_grad)
        } else {
            // Fisher / Ocat path.
            let a_inv = fit.a_inv();
            let xtwx_f = build_xtwx(&self.x_design, &fit.working_weights);
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &fit.working_weights,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_f, dev_grad)
        };

        let ctx = HessIftCtx {
            s_list: &self.s_list,
            lambda: &lambda,
            beta: &fit.beta,
            a_inv: &a_inv_owned,
            xtwx: &xtwx_owned,
            sigma2,
            dev_grad_beta: &dev_grad_beta,
        };
        let h_rho = hess_ift_rho(&ctx);
        for i in 0..n_terms {
            for j in 0..n_terms {
                h[[i, j]] = h_rho[[i, j]];
            }
        }

        // -----------------------------------------------------------------
        // 2) Frozen-β IFT FD for shape rows/cols (n_shape probes, 0 PIRLS).
        // -----------------------------------------------------------------
        // Previous implementation ran `compute_value_grad(θ ± h)` per shape
        // axis, which re-converges PIRLS at the perturbed shape (2·n_shape
        // PIRLS / outer iter). `eval_grad_frozen_beta` returns the IFT
        // analytic gradient at frozen β̂ / μ̂ / η̂ with perturbed family,
        // so the FD now costs only level-1 evaluations + one A_inv·X'·v
        // solve per probe — zero PIRLS. Envelope-theorem analysis: the
        // gradient error from frozen β is O(h), which becomes O(h²) under
        // central FD — the same bound the IFT shape probes against
        // converged β̂ already achieve.
        if n_shape > 0 {
            let eps = 1.0e-4;
            for k in 0..n_shape {
                let mut t_plus = theta.clone();
                let mut t_minus = theta.clone();
                t_plus[n_terms + k] += eps;
                t_minus[n_terms + k] -= eps;
                let g_plus = self.eval_grad_frozen_beta(&t_plus, fit, frozen_ctx)?;
                let g_minus = self.eval_grad_frozen_beta(&t_minus, fit, frozen_ctx)?;
                let col = n_terms + k;
                for j in 0..d {
                    h[[j, col]] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
                }
            }
            // Symmetrise the cross block (n_terms × n_shape) — the FD column
            // for shape axis k carries `H_ρ_k = ∂g_ρ/∂θ_k`; H[i, n_terms+k]
            // was just filled, and H[n_terms+k, i] should equal it.
            for i in 0..n_terms {
                for k in 0..n_shape {
                    let col = n_terms + k;
                    h[[col, i]] = h[[i, col]];
                }
            }
            // Symmetrise within the shape block too (in case the FD probe
            // introduced asymmetry — matches `hess_via_fd_on_grad`'s pass).
            for a in 0..n_shape {
                for b in (a + 1)..n_shape {
                    let r = n_terms + a;
                    let c = n_terms + b;
                    let avg = 0.5 * (h[[r, c]] + h[[c, r]]);
                    h[[r, c]] = avg;
                    h[[c, r]] = avg;
                }
            }
        }

        Ok(h)
    }

    /// **Full Level-2 analytic** REML/LAML Hessian over the joint
    /// `(M + n_shape)` outer-Newton coordinates — port of mgcv_rust
    /// `src/reml/mod.rs::tdist_gdi2_native` (the Hessian half, lines
    /// 1434-1562), translated into gamrs's outer ordering
    /// `[ρ_1, …, ρ_M, θ_1, …, θ_n_shape]` directly so there's no
    /// permutation step.
    ///
    /// Mathematical sketch. At PIRLS-converged β̂ the score is
    /// `S = Dp/(2φ) − ls + ½ log|A| − ½ log|λS|+ − ½·Mp·log(2πφ)` where
    /// `A = X' W X + Σλ_k S_k` and `W = ½·d2_loss_dmu` (mgcv convention).
    /// The total derivative w.r.t. an outer axis `θ` chains through β
    /// via the IFT first-order `b1[k] = ∂β/∂θ_k`:
    ///   - **λ axis**: `b1[k] = −λ_k · A⁻¹ S_k β`
    ///   - **shape axis**: `b1[k] = −½ · A⁻¹ X' Dmuth[k]`
    /// `η1[k] = X·b1[k]` is the linear-predictor sensitivity per axis,
    /// `a1[k] = ∂A/∂θ_k` the matrix sensitivity, and `b2[i,k]` /
    /// `eta2[i,k]` / `a2[i,k]` the matched second-order quantities (with
    /// `b2` solved against the same `A`). The Hessian assembles as
    /// `hess[i,k] = ½·(d2[i,k] + p2[i,k]) − ls2[i,k] + ½·ldet2[i,k]`,
    /// each component a Level-1 / Level-2 chain — see comments inline.
    ///
    /// PIRLS economy: **0 inner solves** (the converged fit's A_inv is
    /// reused; b2 is solved via the same factorisation). One pass over
    /// the `(M + n_shape)²` upper-triangular Hessian entries.
    ///
    /// **A convention**: uses `fit.a_factor` (Fisher A for use_newton_irls
    /// = false families, Newton A otherwise) to match the existing
    /// `analytic_shape_grad_via_ift` IFT gradient. This is the same
    /// matrix the score formula's `log|H|` differentiates, so the
    /// Hessian and gradient share a single A throughout.
    #[allow(clippy::too_many_arguments)]
    fn hess_via_ift_level2(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        family: &crate::family::Family<L, K, V>,
        lv1: &crate::traits::Level1ShapeDerivs,
        lv2: &crate::traits::Level2ShapeDerivs,
        n_shape: usize,
    ) -> Result<Array2<f64>> {
        use ndarray::s;
        let n_terms = self.s_list.len();
        let ntot = n_terms + n_shape;
        debug_assert_eq!(theta.len(), ntot);
        let n = fit.n;
        let p = fit.p;

        // ---- Pick A_inv (Newton-A for use_newton_irls families, else
        // the Fisher A from the converged fit). Mirrors
        // `analytic_shape_grad_via_ift` — keeps grad/Hess consistent.
        let rho_slice: Vec<f64> = theta.slice(s![..n_terms]).to_vec();
        let lambda: Vec<f64> = rho_slice.iter().map(|&r| r.exp()).collect();
        let use_newton = family.loss.use_newton_irls();
        let lazy_tk = if use_newton {
            let prior_w = self
                .prior_weights
                .clone()
                .unwrap_or_else(|| Array1::ones(n));
            let rho_arr = Array1::from(rho_slice.clone());
            let s_total = crate::design::combined_s(&self.s_list, &rho_arr, self.x_design.ncols());
            crate::inner::pirls::lazy_tk_kkt_inputs(
                family,
                &self.y,
                &fit.mu,
                &fit.beta,
                &prior_w,
                &self.x_design,
                &self.s_list,
                &s_total,
                &rho_arr,
            )
        } else {
            None
        };
        let a_inv: Array2<f64> = if let Some(ref tk) = lazy_tk {
            tk.a_newton_inv.clone()
        } else {
            fit.a_inv()
        };

        // ---- η-coord (vs μ-coord) Level-1 derivs. For identity link the
        // factors collapse to identity. We mirror `analytic_shape_grad_via_ift`
        // verbatim so the same A and the same Level-1 transformations are
        // shared between grad and Hessian.
        // mapv-style construction (one trait-method call per row, no
        // index-into-Array1 bounds-check chain). Compiler specialises
        // for identity link → constants → autovectorisation friendly.
        let ig1: Array1<f64> = fit.mu.mapv(|mu_i| {
            let gp = self.family_base.link.d_link_dmu(mu_i);
            if gp.abs() < 1e-300 {
                0.0
            } else {
                1.0 / gp
            }
        });
        let g2g: Array1<f64> = fit.mu.mapv(|mu_i| {
            let gp = self.family_base.link.d_link_dmu(mu_i);
            if gp.abs() < 1e-300 {
                0.0
            } else {
                self.family_base.link.d2_link_dmu(mu_i) / (gp * gp)
            }
        });
        let g3g: Array1<f64> = fit.mu.mapv(|mu_i| {
            let gp = self.family_base.link.d_link_dmu(mu_i);
            if gp.abs() < 1e-300 {
                0.0
            } else {
                self.family_base.link.d3_link_dmu(mu_i) / (gp * gp * gp)
            }
        });
        let prior_w_view = self.prior_weights.as_ref().map(|w| w.view());
        let dmu_arr: Array1<f64> = Array1::from_shape_fn(n, |i| {
            let wt_i = prior_w_view.as_ref().map(|w| w[i]).unwrap_or(1.0);
            wt_i * family.loss.d_loss_dmu(self.y[i], fit.mu[i])
        });
        let dmu2_arr: Array1<f64> = Array1::from_shape_fn(n, |i| {
            let wt_i = prior_w_view.as_ref().map(|w| w[i]).unwrap_or(1.0);
            wt_i * family.loss.d2_loss_dmu(self.y[i], fit.mu[i])
        });

        // Identity-link short-circuits — the full η-coord chain reduces to
        // μ-coord here. Non-identity links would need the `Detath` /
        // `Deta2th` / `Deta3` / `Deta4` cascade (mgcv R `gam.fit4.r:47-78`);
        // gamrs's `analytic_shape_grad_via_ift` builds them but TDist
        // (the only Level-2 family today) is identity-link, so we don't
        // gate on it yet. When NegBin / Ocat Level-2 ports land, swap the
        // identity-only forms below for the general η-chain.
        let is_identity = ig1.iter().all(|&x| (x - 1.0).abs() < 1e-12)
            && g2g.iter().all(|&x| x.abs() < 1e-12)
            && g3g.iter().all(|&x| x.abs() < 1e-12);
        debug_assert!(
            is_identity,
            "hess_via_ift_level2 currently assumes identity link \
             (TDist is the only Level-2 family — wire η-coord chain when \
             NegBin / Ocat land)."
        );

        // ---- b1[k]:  k=0..M-1 → ρ_k axis,  k=M..M+n_shape-1 → shape (k-M).
        // Layout: b1 is (p × ntot), column k is ∂β/∂axis_k.
        let mut b1 = Array2::<f64>::zeros((p, ntot));
        for k in 0..n_terms {
            // ρ axis: b1 = -λ_k · A_inv · S_k · β
            let s_k_beta: Array1<f64> = self.s_list[k].dot(&fit.beta);
            let ainv_sk_beta: Array1<f64> = a_inv.dot(&s_k_beta);
            let lam = lambda[k];
            for r in 0..p {
                b1[[r, k]] = -lam * ainv_sk_beta[r];
            }
        }
        for kk in 0..n_shape {
            // shape axis: b1 = -0.5 · A_inv · X' · Dmuth[:, kk]
            let dmuth_kk: Array1<f64> = lv1.dmuth.column(kk).to_owned();
            let xt_v: Array1<f64> = self.x_design.t().dot(&dmuth_kk);
            let ainv_v: Array1<f64> = a_inv.dot(&xt_v);
            let col = n_terms + kk;
            for r in 0..p {
                b1[[r, col]] = -0.5 * ainv_v[r];
            }
        }
        // η1[k] = X · b1[k] (n × ntot). One matvec per axis instead of
        // a per-row dot product — ndarray's .dot() uses contiguous BLAS-
        // friendly iteration even at small p.
        let mut eta1 = Array2::<f64>::zeros((n, ntot));
        for k in 0..ntot {
            let b1k = b1.column(k).to_owned();
            let eta1_k = self.x_design.dot(&b1k);
            eta1.column_mut(k).assign(&eta1_k);
        }

        // ---- s_beta_total = (Σ λ_j S_j) · β — reused throughout p1, p2.
        let mut s_beta_total = Array1::<f64>::zeros(p);
        for j in 0..n_terms {
            let s_j_beta = self.s_list[j].dot(&fit.beta);
            for r in 0..p {
                s_beta_total[r] += lambda[j] * s_j_beta[r];
            }
        }

        // ---- a1[k]: ∂A/∂axis_k — stored as a single weight vector w1[k]
        // representing the diagonal of X' diag(w1[k]) X plus a separate
        // penalty contribution for ρ axes. We never materialise the dense
        // a1; instead carry (w1, lam_penalty_idx) and compute traces on
        // the fly.
        //   ρ axis k:    a1[k] = X' diag(0.5·dmu3·η1[k]) X + λ_k·S_k
        //   shape k:     a1[k] = X' diag(0.5·dmu3·η1[k] + 0.5·Dmu2th[:, k-M]) X
        // We accumulate the LEVERAGE diag(X·A_inv·X') = h_diag and the
        // factored A_inv to compute tr(A_inv · a1) and tr((A_inv·a1)·(A_inv·a1')).
        let dmu3 = &lv1.dmu3;
        let mut w1 = Array2::<f64>::zeros((n, ntot)); // per-axis diag-weights
        for k in 0..n_terms {
            // w1[:, k] = 0.5 · dmu3 · eta1[:, k]  (broadcast).
            let col = 0.5 * dmu3 * eta1.column(k);
            w1.column_mut(k).assign(&col);
        }
        for kk in 0..n_shape {
            let col = n_terms + kk;
            for i in 0..n {
                w1[[i, col]] = 0.5 * dmu3[i] * eta1[[i, col]] + 0.5 * lv1.dmu2th[[i, kk]];
            }
        }

        // ---- Precompute A_inv · A_1[k]  (p × p) per axis (used by ldet2).
        // For ρ axis k, A_1[k] includes the λ_k·S_k term; that's a separate
        // contribution: A_inv·(X'·diag(w1)·X + λ_k·S_k).
        // To avoid n_shape+M dense p×p matrices, we accumulate the trace
        // pieces lazily. ldet2 needs tr((A_inv·a1[i])·(A_inv·a1[k])) —
        // we build a_inv_xt_w (p × n) on the fly per axis (k) and reuse.
        // `xt` is a *view* (Array2::t() returns ArrayView2; no copy).
        // ndarray's `.dot()` accepts views, so the downstream
        // `xt.dot(...)` sites are unchanged. Removing the previous
        // `.to_owned()` saves one O(n·p) transpose copy (~160 KB at
        // n=2000, p=10) per Hessian assembly call — measured ~20 μs
        // saving per call by the `profile` feature's phase timers.
        let xt = self.x_design.t();
        let mut ai_a1: Vec<Array2<f64>> = Vec::with_capacity(ntot);
        for k in 0..ntot {
            // A_1[k] = X' diag(w1[:, k]) X + (λ_k·S_k if k < M)
            // Build wx = w · X via broadcast (`w.insert_axis(1)` is
            // (n, 1) broadcasting against X's (n, p) shape).
            let w_col = w1.column(k);
            let wx: Array2<f64> = &self.x_design * &w_col.insert_axis(ndarray::Axis(1));
            let mut a1_k = xt.dot(&wx);
            if k < n_terms {
                let lam = lambda[k];
                a1_k = a1_k + lam * &self.s_list[k];
            }
            ai_a1.push(a_inv.dot(&a1_k));
        }

        // ---- ls1, ls2 (saturated-LL gradient + Hessian over θ-shape only).
        // Both are zero on ρ axes — ls is λ-independent.
        let sum_dls = family.loss.sum_saturated_log_lik_dtheta(
            self.y.view(),
            1.0,
            self.prior_weights.as_ref().map(|w| w.view()),
        );
        let sum_d2ls = family.loss.sum_saturated_log_lik_d2theta(
            self.y.view(),
            1.0,
            self.prior_weights.as_ref().map(|w| w.view()),
        );
        debug_assert_eq!(sum_dls.len(), n_shape);
        debug_assert_eq!(sum_d2ls.len(), n_shape * (n_shape + 1) / 2);
        let mut ls2_full = Array2::<f64>::zeros((ntot, ntot));
        for a in 0..n_shape {
            for b in a..n_shape {
                let v = sum_d2ls[shape_pair_index(a, b, n_shape)];
                let r = n_terms + a;
                let c = n_terms + b;
                ls2_full[[r, c]] = v;
                ls2_full[[c, r]] = v;
            }
        }

        // ---- Main symmetric loop: for each (i, k) with i ≤ k, fill the
        // Hessian entry. Mirrors `tdist_gdi2_native` lines 1441-1551 with
        // index ranges adjusted to gamrs's `[ρ; θ]` layout.
        let mut hess = Array2::<f64>::zeros((ntot, ntot));
        for i in 0..ntot {
            for k in i..ntot {
                // ── Build RHS for b2[i,k] solve ──────────────────────
                // mgcv: rhs_w starts as -det3 · η1[i] · η1[k] (n-vector).
                // Then μ-cross-θ corrections: -Dmuth[k]·η1[i] (if k shape)
                // and -Dmuth[i]·η1[k] (if i shape).
                //
                // **Broadcast-expression form** instead of indexed for-loops.
                // ndarray's `&a * &b * &c` compiles to SIMD-friendly
                // element-wise iteration without bounds-check overhead per
                // step. The mgcv_rust port uses this exact pattern at
                // `nn_exploring/src/reml/mod.rs:1443`.
                let eta1_i = eta1.column(i);
                let eta1_k = eta1.column(k);
                // dmu3 is `&Array1<f64>`; -& doesn't apply to a reference,
                // so use `-1.0 * &dmu3` to negate inline.
                let mut rhs_w: Array1<f64> = (-1.0 * dmu3) * eta1_i * eta1_k;
                if k >= n_terms {
                    let kk = k - n_terms;
                    let dmuth_k = lv1.dmuth.column(kk);
                    rhs_w = rhs_w - &dmuth_k * &eta1_i;
                }
                if i >= n_terms {
                    let ii = i - n_terms;
                    let dmuth_i = lv1.dmuth.column(ii);
                    rhs_w = rhs_w - &dmuth_i * &eta1_k;
                }
                // rhs (length p) = X' · rhs_w
                let mut rhs: Array1<f64> = xt.dot(&rhs_w);
                // Penalty contributions to RHS:
                //   if k is ρ: -2 λ_k · S_k · b1[i]
                //   if i is ρ: -2 λ_i · S_i · b1[k]
                //   if i == k AND ρ: extra -2 λ_i · S_i · β
                if k < n_terms {
                    let s_k_b1_i: Array1<f64> = self.s_list[k].dot(&b1.column(i).to_owned());
                    let lam = lambda[k];
                    for r in 0..p {
                        rhs[r] -= 2.0 * lam * s_k_b1_i[r];
                    }
                }
                if i < n_terms {
                    let s_i_b1_k: Array1<f64> = self.s_list[i].dot(&b1.column(k).to_owned());
                    let lam = lambda[i];
                    for r in 0..p {
                        rhs[r] -= 2.0 * lam * s_i_b1_k[r];
                    }
                }
                if i == k && i < n_terms {
                    // mgcv `tdist_gdi2_native:1467-1469`: extra -2λ_i·S_i·β
                    // for the diagonal ρ entry (catches the second λ in
                    // ρ_i's RHS — `2·λ_i·S_i·β` total counted twice via b1[i]
                    // already gives one; the diagonal needs one more).
                    let s_i_beta: Array1<f64> = self.s_list[i].dot(&fit.beta);
                    let lam = lambda[i];
                    for r in 0..p {
                        rhs[r] -= 2.0 * lam * s_i_beta[r];
                    }
                }
                // Level-2 shape×shape cross term: -X' · Dmu_th2[pair]
                if i >= n_terms && k >= n_terms {
                    let ii = i - n_terms;
                    let kk = k - n_terms;
                    let pair = shape_pair_index(ii.min(kk), ii.max(kk), n_shape);
                    let dmu_th2_p: Array1<f64> = lv2.dmu_th2.column(pair).to_owned();
                    let xt_v: Array1<f64> = xt.dot(&dmu_th2_p);
                    for r in 0..p {
                        rhs[r] -= xt_v[r];
                    }
                }
                // b2[i,k] = 0.5 · A_inv · rhs
                let b2_ik: Array1<f64> = a_inv.dot(&rhs);
                let b2_ik: Array1<f64> = &b2_ik * 0.5;
                // η2[i,k] = X · b2[i,k]
                let eta2_ik: Array1<f64> = self.x_design.dot(&b2_ik);

                // ── d2[i,k]: second deriv of D ────────────────────────
                // Broadcast-expr form — see rhs_w build comment above for
                // why this beats indexed for-loops.
                let mut d2_ik = (&dmu2_arr * &eta1_i * eta1_k).sum() + (&dmu_arr * &eta2_ik).sum();
                if i >= n_terms && k >= n_terms {
                    let ii = i - n_terms;
                    let kk = k - n_terms;
                    let pair = shape_pair_index(ii.min(kk), ii.max(kk), n_shape);
                    d2_ik += lv2.dth2.column(pair).sum();
                }
                if i >= n_terms {
                    let ii = i - n_terms;
                    d2_ik += (&lv1.dmuth.column(ii) * &eta1_k).sum();
                }
                if k >= n_terms {
                    let kk = k - n_terms;
                    d2_ik += (&lv1.dmuth.column(kk) * &eta1_i).sum();
                }

                // ── p2[i,k]: second deriv of P = β'(Σ λ S)β ──────────
                // p2 = 2·b2'·s_beta_total + 2·b1[i]'·(Σ λ_m S_m · b1[k])
                //      + extra λ terms when i or k is ρ.
                let mut p2_ik = 0.0_f64;
                for r in 0..p {
                    p2_ik += 2.0 * b2_ik[r] * s_beta_total[r];
                }
                // 2 · b1[i]' · (Σ λ_m S_m · b1[k])
                let mut s_b1_k = Array1::<f64>::zeros(p);
                for m in 0..n_terms {
                    let s_m_b1_k: Array1<f64> = self.s_list[m].dot(&b1.column(k).to_owned());
                    for r in 0..p {
                        s_b1_k[r] += lambda[m] * s_m_b1_k[r];
                    }
                }
                for r in 0..p {
                    p2_ik += 2.0 * b1[[r, i]] * s_b1_k[r];
                }
                if k < n_terms {
                    let s_k_beta: Array1<f64> = self.s_list[k].dot(&fit.beta);
                    let lam = lambda[k];
                    let mut acc = 0.0_f64;
                    for r in 0..p {
                        acc += b1[[r, i]] * s_k_beta[r];
                    }
                    p2_ik += 2.0 * lam * acc;
                }
                if i < n_terms {
                    let s_i_beta: Array1<f64> = self.s_list[i].dot(&fit.beta);
                    let lam = lambda[i];
                    let mut acc = 0.0_f64;
                    for r in 0..p {
                        acc += b1[[r, k]] * s_i_beta[r];
                    }
                    p2_ik += 2.0 * lam * acc;
                }
                if i == k && i < n_terms {
                    let s_i_beta: Array1<f64> = self.s_list[i].dot(&fit.beta);
                    let lam = lambda[i];
                    let mut acc = 0.0_f64;
                    for r in 0..p {
                        acc += fit.beta[r] * s_i_beta[r];
                    }
                    p2_ik += lam * acc;
                }

                // ── ldet2[i,k]: ½ ∂² log|A| via trace identity.
                //   ldet2 = tr(A_inv · a2[i,k]) - tr((A_inv·a1[i])·(A_inv·a1[k]))
                // Build w2[r] = ∂²(W's diag)/(∂θ_i ∂θ_k) — broadcast-expr form.
                let mut w2: Array1<f64> = &lv2.dmu4 * &eta1_i * eta1_k + dmu3 * &eta2_ik;
                if i >= n_terms {
                    let ii = i - n_terms;
                    w2 = w2 + &lv2.dmu3_th.column(ii) * &eta1_k;
                }
                if k >= n_terms {
                    let kk = k - n_terms;
                    w2 = w2 + &lv2.dmu3_th.column(kk) * &eta1_i;
                }
                if i >= n_terms && k >= n_terms {
                    let ii = i - n_terms;
                    let kk = k - n_terms;
                    let pair = shape_pair_index(ii.min(kk), ii.max(kk), n_shape);
                    w2 = w2 + lv2.dmu2_th2.column(pair);
                }
                // a2[i,k] = X' · diag(0.5·w2) · X  (+ λ_i·S_i if diagonal ρ).
                // wx2 built via broadcast (n,1)·(n,p) — same pattern as
                // the a1_loop above.
                let half_w2 = &w2 * 0.5;
                let wx2: Array2<f64> =
                    &self.x_design * &half_w2.view().insert_axis(ndarray::Axis(1));
                let mut a2_ik = xt.dot(&wx2);
                if i == k && i < n_terms {
                    let lam = lambda[i];
                    a2_ik = a2_ik + lam * &self.s_list[i];
                }
                // tr(A_inv · a2[i,k])
                let ai_a2 = a_inv.dot(&a2_ik);
                let tr_term1: f64 = ai_a2.diag().sum();
                // tr((A_inv·a1[i])·(A_inv·a1[k])) — symmetric matrix prod.
                let tr_term2: f64 = (&ai_a1[i] * &ai_a1[k].t()).sum();
                let ldet2_ik = tr_term1 - tr_term2;

                // ── Assemble Hessian entry ────────────────────────────
                let v = 0.5 * (d2_ik + p2_ik) - ls2_full[[i, k]] + 0.5 * ldet2_ik;
                hess[[i, k]] = v;
                hess[[k, i]] = v;
            }
        }

        // Suppress unused-warning for `sum_dls` — kept here so the
        // signature is parallel to the gradient path (future Level-2
        // families may need a Hessian-side adjustment that references it).
        let _ = sum_dls;

        Ok(hess)
    }
}
