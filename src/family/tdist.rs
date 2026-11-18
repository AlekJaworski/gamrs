//! Scaled-t (TDist / mgcv `scat`) with identity link.

use crate::traits::{Loss, VarianceFn};

use super::link::IdentityLink;
use super::Family;

/// Minimum degrees of freedom for the ν reparameterisation `ν = MIN_DF +
/// exp(θ₁)`. Matches mgcv `scat(min.df = 3)` (R `efam.r:1248`). The shape
/// transform `θ₁ = log(ν − MIN_DF)` floors ν at MIN_DF during estimation;
/// the chain-rule factor `dν/dθ₁ = ν − MIN_DF` (mgcv's `nu2 = nu − min.df`)
/// permeates every ν-derivative below, so this constant — NOT 2 — defines
/// the ν-axis geometry the outer Newton sees. (mgcv's earlier default was 2;
/// it was raised to 3 because "low df and low variance promotes
/// indefiniteness", `efam.r:1419`.)
pub const MIN_DF: f64 = 3.0;

/// Heavy-tailed scaled-t likelihood: `y_i ~ μ_i + σ · T_ν` where `T_ν` is
/// a standard Student-t with `ν` degrees of freedom. Used for robust
/// regression where Gaussian noise mis-specifies the tails.
///
/// **Stateful loss** — `nu` and `sigma2` are SHAPE PARAMETERS of the
/// family, not data. In mgcv they're jointly optimised with the smoothing
/// parameter λ via an outer Newton over `[log λ, log σ², ν-transform]`.
///
/// Phase 2a ships TDist with `nu` / `sigma2` as struct fields fixed at
/// construction time. Phase 2b will extend the outer optimiser to handle
/// multi-θ so the shape params can be joint-optimised (see
/// architecture-assumptions.md §E for the plan).
#[derive(Clone)]
pub struct TDist {
    /// Degrees of freedom. mgcv requires ν > 2 for finite variance; we
    /// don't enforce here (PIRLS handles ν ∈ (1, 2] in principle, just
    /// with slow tails).
    pub nu: f64,
    /// Squared scale parameter. The actual t-scale is √σ². Plays the same
    /// role as Gaussian σ² but is internal to the family — mgcv's
    /// dispersion `scale` stays at 1 for scat.
    pub sigma2: f64,
}

/// Variance function for TDist (location-scale family). The "variance" is
/// constant `ν·σ²/(ν-2)` for finite ν > 2, OR equivalently mgcv treats it
/// as just `σ²` and folds `ν/(ν-2)` into the working weights. gamrs mirrors
/// mgcv's convention: `V(μ) = σ²` and the PIRLS working weights use the
/// t-specific `∂²L/∂μ²` directly via `Loss::d2_loss_dmu`.
#[derive(Clone)]
pub struct TVariance {
    pub sigma2: f64,
}

impl Loss for TDist {
    /// `D_i = (ν+1) · log(1 + (y-μ)² / (ν·σ²))` per mgcv `scat$dev.resids`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let r = y - mu;
        (self.nu + 1.0) * (1.0 + r * r / (self.nu * self.sigma2)).ln()
    }

    /// Saturated log-lik per observation: `log Γ((ν+1)/2) - log Γ(ν/2)
    /// - 0.5 log(π·ν·σ²)` — independent of y (scat is location-scale, so
    /// the saturated density is constant in the response). The `_scale`
    /// arg is the external dispersion — TDist owns its scale via the
    /// shape param `self.sigma2`, so the external one is ignored.
    ///
    /// **Why both Γ terms matter under joint Newton on (λ, ν, σ²)**:
    /// historically Phase-2a dropped the Γ ratio with the rationale "Γ
    /// terms are constants in (ν, σ²)" — that is **false**: Γ((ν+1)/2)
    /// and Γ(ν/2) both move with ν, and the Σ_i ls_i term carries an
    /// n·(dlog Γ((ν+1)/2)/dν − dlog Γ(ν/2)/dν) component into the
    /// LAML gradient w.r.t. log(ν - 2). Dropping it (as Phase 2a did)
    /// made the outer Newton's ∂LAML/∂(log(ν - 2)) chase the wrong
    /// optimum, pulling ν toward the lower bound (ν → 2⁺ saturated at
    /// `log(ν - 2) = -10` on the multi-smooth synthetic) instead of
    /// the interior optimum mgcv finds at ν ≈ 5. Includes both Γ terms
    /// to match v0.x `pirls/mod.rs:521-528` (Family::TDist branch of
    /// `saturated_log_likelihood`) byte-for-byte at fixed (ν, σ²).
    fn saturated_log_lik(&self, _y: f64, _scale: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let half_nu_p1 = (self.nu + 1.0) / 2.0;
        let half_nu = self.nu / 2.0;
        crate::special::log_gamma(half_nu_p1)
            - crate::special::log_gamma(half_nu)
            - 0.5 * (pi * self.nu * self.sigma2).ln()
    }

    /// `∂D/∂μ = -2(ν+1)·(y-μ) / (ν·σ² + (y-μ)²)`.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let r = y - mu;
        -2.0 * (self.nu + 1.0) * r / (self.nu * self.sigma2 + r * r)
    }

    /// `∂²D/∂μ² = 2(ν+1)·(ν·σ² - r²) / (ν·σ² + r²)²` where `r = y - μ`.
    /// Positive for `|r| < √(ν·σ²)` (the "core" of the distribution) and
    /// negative for outliers — this is what gives scat its robustness.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let r = y - mu;
        let denom = self.nu * self.sigma2 + r * r;
        2.0 * (self.nu + 1.0) * (self.nu * self.sigma2 - r * r) / (denom * denom)
    }

    /// Dispersion φ = 1 for scat — the actual scale lives inside the
    /// family as `sigma2`. Same convention as mgcv.
    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0)
    }

    /// scat owns 2 shape params: `[log σ², log(ν − MIN_DF)]` with
    /// `MIN_DF = 3` (mgcv `scat(min.df = 3)`). The `log(ν − MIN_DF)`
    /// transform is mgcv's choice (`gam.fit5.r`) and floors ν at MIN_DF.
    fn n_shape_params(&self) -> usize {
        2
    }
    /// mgcv `build_outer_search_vector`: TDistLogSigma2 step cap 1.0,
    /// TDistLogNu (log(ν − MIN_DF)) step cap 1.0.
    fn shape_axis_step_caps(&self) -> Vec<f64> {
        vec![1.0, 1.0]
    }

    /// Per-axis (lo, hi) bounds for `θ = [log σ², log(ν − MIN_DF)]`.
    ///
    /// The trait default clamps **every** shape axis to `[-10, 10]`. That
    /// is right for a dimensionless axis like ν's `log(ν − 2)` (ν ≈ 5 at
    /// the interior optimum), but **wrong for `log σ²`**: σ² is the t-scale
    /// in the units of `y²`, so its optimum tracks the data scale. For a
    /// response with `Var(y) ≈ 3e11` (e.g. house prices) the REML optimum
    /// sits near `log σ² ≈ 23`, far outside `[-10, 10]`. With the default
    /// clamp the joint `(ρ, log σ², log(ν−2))` Newton pins `log σ²` at the
    /// upper bound `10` (`σ² = e¹⁰ ≈ 2.2e4`), the fit never reaches the
    /// data scale, the outer Newton reports non-convergence, and `scat`
    /// degenerates toward the Gaussian fit instead of down-weighting the
    /// high-price tail — the exact symptom in `data/SCAT_PARITY_BUG.md`.
    ///
    /// mgcv (and the reference `mgcv_rust`, `pirls/mod.rs:1535/1605`)
    /// profile σ² freely with no bound (a sample-variance-initialised MLE
    /// update inside PIRLS). We widen the σ² axis to the same effectively
    /// unbounded `[-50, 50]` the ρ axes use (`σ² ∈ [2e-22, 5e21]`, covering
    /// any realistic data scale) and keep ν on the default `[-10, 10]`.
    fn shape_axis_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-50.0, 50.0), (-10.0, 10.0)]
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(params.len(), 2, "TDist expects 2 shape params");
        self.sigma2 = params[0].exp();
        self.nu = MIN_DF + params[1].exp();
    }
    fn get_shape_params(&self) -> Vec<f64> {
        vec![self.sigma2.ln(), (self.nu - MIN_DF).ln()]
    }

    /// Match v0.x `fit_pirls_tdist`'s β-tolerance. v0.x's scat diagnostic
    /// call site (`lib.rs:1162`) and the regular outer-fit path both pass
    /// `1e-8` to `fit_pirls_tdist`. gamrs's PirlsOpts default of `1e-9`
    /// stops one decimal later than v0.x, leaving a residual β-gap that
    /// flows through Layer-3 into the score-formula's `log|H|`. Same
    /// convention as ocat (commit `4c95a72`).
    fn pirls_dev_rel_tol(&self) -> f64 {
        1.0e-8
    }

    // NOTE on `score_rank_adjustment`: TDist keeps the trait default `0` —
    // the mathematically correct positive-eigenvalue count from
    // `rank_and_log_pseudo_det`. An `-1` override rode in on an unrelated
    // tensor-dispatch commit (`fa3df55`, labelled EXPERIMENTAL) and stayed
    // for three months. It is not a free constant: `log|λS|₊` contributes
    // `-½·rank·ρ` to the score, so dropping the rank by one tilts the whole
    // REML surface by `+ρ/2` per smooth and the outer Newton converges
    // under-penalised. Measured on a real `garage_spaces` term (620
    // rows, 5 distinct x, k=5, so the basis is saturated): mgcv's own
    // fixed-sp REML sweep bottoms out at edf 2.37, gamrs with the `-1`
    // landed on edf 4.02 — mgcv's own answer at ~30× less penalty — and
    // adding ½ρ back to gamrs' sweep moved its minimum onto mgcv's to
    // within 0.04 REML units. Every scat parity fixture tightened 8-16×
    // when it came out. Do not reintroduce it without a fixed-sp sweep
    // showing mgcv agrees.

    /// mgcv `scat` initialize starts μ at the response (`mustart <- y`,
    /// `efam.r:1430`), so the first PIRLS residuals are ~0 — the natural
    /// saturated start for the location-scale t. The trait default `(y+ȳ)/2`
    /// instead injects an artificial half-shrink-to-mean residual on iteration
    /// 0. Identity link, so μ is on the (internally standardized) response
    /// scale; exact-zero rows get mgcv's `+0.1` nudge.
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        y.mapv(|yi| if yi == 0.0 { 0.1 } else { yi })
    }

    /// **Observed-W PIRLS pair for scat** — port of mgcv R's `gam.fit4.r`
    /// inner-loop W/z build (lines 368-399). Direct port of `0.5·D_μμ·(dμ/dη)²`
    /// with `(y − μ) · g'(μ)·dμ/dη / α` working response, expected-Hessian
    /// fallback when `D_μμ ≤ 0`.
    ///
    /// For TDist + identity link the derivatives reduce to:
    ///   `W_obs = ½·d²L/dμ² = (ν+1)·(νσ² − r²) / (νσ² + r²)²`
    ///   `z = η − D'_μ / D''_μ = η + r·s / (νσ² − r²)`,  `s = νσ² + r²`
    /// (the `D'/D''` ratio is the Newton direction along μ). When
    /// `D_μμ ≤ 0` (heavy-tail outlier: `|r| > √(νσ²)`) substitute the
    /// expected curvature `E[D_μμ] = (ν+1)/((ν+3)σ²)` and the Fisher
    /// working response `z = η + (y − μ)·g'(μ) = η + r` — matches
    /// `gam.fit4.r:392-399`.
    ///
    /// This aligns gamrs PIRLS for TDist with mgcv R's `scat`. Without it,
    /// gamrs PIRLS runs Fisher `W = prior_w/σ²` (constant in μ) — fine for
    /// β̂ convergence (Fisher and observed share the same fixed point at
    /// PIRLS convergence) but the resulting `log|A_F|` is a **different**
    /// function of `(σ², ν)` than `log|A_obs|` that mgcv R's `gdi2`
    /// Hessian assembly assumes. Routing TDist through the observed-W
    /// pair makes `fit.a_factor` carry observed A, which feeds straight
    /// into `analytic_shape_grad_via_ift` / `hess_via_ift_level2` — the
    /// IFT chain and Level-2 closed-form Hessian now differentiate the
    /// same A the score's `log|H|` uses.
    fn irls_observed_pair(
        &self,
        y: ndarray::ArrayView1<f64>,
        mu: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: ndarray::ArrayView1<f64>,
    ) -> Option<(ndarray::Array1<f64>, ndarray::Array1<f64>)> {
        use ndarray::Array1;
        let n = y.len();
        debug_assert_eq!(mu.len(), n);
        debug_assert_eq!(eta.len(), n);
        debug_assert_eq!(prior_w.len(), n);
        let nu = self.nu;
        let sigma2 = self.sigma2;
        let q = nu * sigma2;
        // Expected curvature ½·E[D_μμ] (mgcv `Dd`: `EDmu2 = 2(ν+1)/((ν+3)σ²)`
        // at `efam.r:1326`; the IRLS weight is `½·EDmu2`). The PRIOR code used
        // `(ν+1)/(2(ν+3)σ²) = ¼·EDmu2` — a **factor-2 error** vs mgcv.
        let w_exp = (nu + 1.0) / ((nu + 3.0) * sigma2);
        let mut w = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = y[i] - mu[i];
            let r2 = r * r;
            let s = q + r2;
            // mgcv R `gam.fit4.r:368-370` (the `scat`/`Dd` IRLS):
            //   w  = ½·Dmu2,            z = η − Dmu/Dmu2.
            // Dmu  = −2(ν+1)·r / s,     Dmu2 = 2(ν+1)(νσ² − r²) / s².
            // Identity link → dμ/dη = 1, d²μ/dη² = 0, so the η-coord weight
            // collapses to ½·Dmu2.
            let dmu = -2.0 * (nu + 1.0) * r / s;
            let dmu2 = 2.0 * (nu + 1.0) * (q - r2) / (s * s);
            let w_obs = 0.5 * dmu2;
            if w_obs > 1e-12 && w_obs.is_finite() {
                // Core row: observed Newton step.
                w[i] = prior_w[i] * w_obs;
                z[i] = eta[i] - dmu / dmu2;
            } else {
                // Outlier row (`|r| ≥ √(νσ²)` ⇒ observed curvature ≤ 0).
                //
                // The PRIOR code paired the expected weight with the *Fisher*
                // response `z = η + r`, giving working response
                // `W(z − η) = w_exp·r`, whose IRLS fixed point is
                // `λSβ = X'·w_exp·r ≠ X'·(−½·Dmu)` — it pulls outliers TOWARD
                // y (Gaussian-like), destroying scat's robustness and
                // inflating the fit above the Gaussian one
                // (`data/SCAT_PARITY_BUG.md`). The correct response uses the
                // EXPECTED-info Newton step `z = η − Dmu/EDmu2`, i.e.
                // `z = η − ½·Dmu/w_exp`, so `W(z − η) = w_exp·(−½·Dmu/w_exp)
                // = −½·Dmu` — the SAME penalised-deviance gradient term as the
                // core rows. The fixed point is then the true stationary point
                // `λSβ = X'·(−½·Dmu)` (mgcv `gam.fit4.r`: the `wz = Wη − ½·Dmu`
                // form preserves `−½·Dmu` regardless of the weight), while the
                // positive `w_exp` keeps `X'WX + λS` factorisable.
                w[i] = prior_w[i] * w_exp;
                z[i] = eta[i] - 0.5 * dmu / w_exp;
            }
        }
        Some((w, z))
    }

    /// `Σᵢ wt_i · ∂ls_i/∂θ_k` for the two scat shape axes
    /// `θ = [log σ², log(ν − 2)]`. The per-obs ls is y-independent for scat
    /// (location-scale) and equal to
    /// `log Γ((ν+1)/2) − log Γ(ν/2) − 0.5·log(π·ν·σ²)`; its θ-derivatives
    /// are therefore constants across rows, so the row sum is just
    /// `(Σ wt) · ∂ls/∂θ_k`.
    ///
    /// - **log σ²**: `∂ls/∂(log σ²) = -0.5` (only the `-0.5·log σ²` part
    ///   depends on σ²).
    /// - **log(ν − 2)**: with `dν/d(log(ν−2)) = ν − 2`,
    ///   `∂ls/∂(log(ν−2)) = (ν−2)/2·[ψ((ν+1)/2) − ψ(ν/2)] − (ν−2)/(2ν)`
    ///   — matches mgcv_rust's `ls1[0]` block in
    ///   `reml/mod.rs::tdist_gdi2_native` line 1425
    ///   (their native order has `log(ν−2)` at index 0 and `log σ` at 1;
    ///   gamrs reorders to `[log σ², log(ν−2)]`).
    ///
    /// Without this override the trait default ships `vec![0.0; 2]`, which
    /// the IFT analytic shape-gradient consumer subtracts at
    /// `score/shape_aware/gradient.rs::analytic_shape_grad_via_ift` — the
    /// resulting ∂score/∂(shape) at the centre is off by a constant per
    /// axis. Before v0.10.x, `eval_grad_with_fit` fell back to FD-on-value
    /// for TDist (which sees ls implicitly through `score_value`), so the
    /// missing ls-derivatives were masked; routing TDist's centre gradient
    /// through the IFT path exposes it.
    fn sum_saturated_log_lik_dtheta(
        &self,
        y: ndarray::ArrayView1<f64>,
        _scale: f64,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Vec<f64> {
        use crate::special::digamma;
        let nu = self.nu;
        let nu_minus_df = nu - MIN_DF;
        let half_nu_p1 = (nu + 1.0) / 2.0;
        let half_nu = nu / 2.0;
        let nu2nu = nu_minus_df / nu;
        let sum_w: f64 = match prior_w {
            Some(w) => w.iter().sum(),
            None => y.len() as f64,
        };
        let dls_dlog_sigma2 = -0.5;
        let dls_dlog_nu_m2 =
            0.5 * nu_minus_df * (digamma(half_nu_p1) - digamma(half_nu)) - 0.5 * nu2nu;
        vec![sum_w * dls_dlog_sigma2, sum_w * dls_dlog_nu_m2]
    }

    /// Provide Level-1 derivatives (`Dmu3, Dth, Dmuth, Dmu2th`) to the
    /// shape-aware score's analytic θ-gradient assembly. Mirrors ocat's
    /// `OcatLoss::level1_shape_derivatives` (commits `85946a1` + `c38083c`)
    /// — the IFT path in `score/shape_aware.rs::analytic_shape_grad_via_ift`
    /// and the Tk·KK' β-chain in `compute_rho_envelope_gradient` are
    /// family-agnostic; they fire as soon as the loss returns `Some(...)`.
    ///
    /// For scat the two shape params are `θ_0 = log σ²` and
    /// `θ_1 = log(ν − 2)`. All four arrays are analytic, derived from
    /// `D(y, μ; ν, σ²) = (ν+1) · log(1 + (y−μ)²/(ν·σ²))`.
    ///
    /// Notation in the derivation: `r = y − μ`, `q = ν·σ²`, `s = q + r²`.
    /// Identity link so `μ = η`. The shape-transform Jacobians are
    /// `∂σ²/∂θ_0 = σ²`, `∂q/∂θ_0 = q`, `∂(ν−2)/∂θ_1 = ν − 2`,
    /// `∂ν/∂θ_1 = ν − 2`, `∂q/∂θ_1 = σ²·(ν − 2)` (= `qs_theta1` below).
    ///
    /// The `dmu3` / `dth` / `dmuth` / `dmu2th` arrays already incorporate
    /// the per-row prior weight (same convention as ocat —
    /// `family/ocat.rs::ocat_dd_level1`; mgcv `efam.r:2814-2832`).
    fn level1_shape_derivatives(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<crate::traits::Level1ShapeDerivs> {
        use ndarray::{Array1, Array2};
        let n = y.len();
        let nu = self.nu;
        let sigma2 = self.sigma2;
        let nu_p1 = nu + 1.0;
        let nu_minus_df = nu - MIN_DF;
        let qs_theta1 = sigma2 * nu_minus_df; // ∂q/∂θ_1
        let q = nu * sigma2; // ν·σ² — constant across rows

        let mut dmu3 = Array1::<f64>::zeros(n);
        let mut dth = Array2::<f64>::zeros((n, 2));
        let mut dmuth = Array2::<f64>::zeros((n, 2));
        let mut dmu2th = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            let r = y[i] - eta[i];
            let r2 = r * r;
            let s = q + r2;
            let s2 = s * s;
            let s3 = s2 * s;
            let wt_i = prior_w.map(|w| w[i]).unwrap_or(1.0);

            // ∂³D/∂μ³ = 4·r·(ν+1)·(3q − r²) / s³. Includes wt (ocat
            // convention — IFT consumer pre-applies wt at this step).
            dmu3[i] = wt_i * 4.0 * r * nu_p1 * (3.0 * q - r2) / s3;

            // ── θ_0 = log σ² ────────────────────────────────────────────
            // ∂D/∂θ_0 = (ν+1)·(q/s − 1) = −(ν+1)·r² / s.
            dth[[i, 0]] = wt_i * (-nu_p1 * r2 / s);
            // ∂(∂D/∂μ)/∂θ_0 = 2(ν+1)·r·q / s².
            dmuth[[i, 0]] = wt_i * (2.0 * nu_p1 * r * q / s2);
            // ∂(∂²D/∂μ²)/∂θ_0 = 2(ν+1)·q·(3r² − q) / s³.
            dmu2th[[i, 0]] = wt_i * (2.0 * nu_p1 * q * (3.0 * r2 - q) / s3);

            // ── θ_1 = log(ν − 2) ────────────────────────────────────────
            // ∂D/∂θ_1 = (ν−2)·[log(1 + r²/q) − (ν+1)·r²/(ν·s)].
            let log_term = if q > 0.0 { (1.0 + r2 / q).ln() } else { 0.0 };
            dth[[i, 1]] = wt_i * nu_minus_df * (log_term - nu_p1 * r2 / (nu * s));
            // ∂(∂D/∂μ)/∂θ_1 = −2r·[(ν−2)·s − (ν+1)·qs_theta1] / s².
            dmuth[[i, 1]] = wt_i * (-2.0 * r * (nu_minus_df * s - nu_p1 * qs_theta1) / s2);
            // ∂(∂²D/∂μ²)/∂θ_1
            //   = 2·[(ν−2)·(q − r²)·s + (ν+1)·qs_theta1·(3r² − q)] / s³.
            dmu2th[[i, 1]] = wt_i
                * (2.0 * (nu_minus_df * (q - r2) * s + nu_p1 * qs_theta1 * (3.0 * r2 - q)) / s3);
        }

        Some(crate::traits::Level1ShapeDerivs {
            dmu3,
            dth,
            dmuth,
            dmu2th,
        })
    }

    fn has_observed_curvature(&self) -> bool {
        true
    }

    /// Expected (Fisher) curvature — mgcv's `wf` for scat, and the weight edf
    /// and vcov must be built from. `efam.r:1327-1329` sets
    /// `EDmu2 = 2(ν+1)/((ν+3)σ²)` on every row, so `½·EDmu2` is a single
    /// scalar. Same value `irls_observed_pair` substitutes on outlier rows —
    /// but here it applies to ALL rows, which is the difference that matters.
    fn expected_curvature_weights(
        &self,
        y: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<ndarray::Array1<f64>> {
        use ndarray::Array1;
        let w_exp = (self.nu + 1.0) / ((self.nu + 3.0) * self.sigma2);
        Some(Array1::from_shape_fn(y.len(), |i| {
            prior_w.map(|p| p[i]).unwrap_or(1.0) * w_exp
        }))
    }

    /// Observed curvature `½·D_μμ` on EVERY row, negatives included — the
    /// bracket in `∂β/∂ρ_j = −λ_j·[X'diag(½·D_μμ)X + Σλ_kS_k]⁻¹·S_j·β`.
    ///
    /// Deliberately WITHOUT `irls_observed_pair`'s expected-curvature
    /// substitution: that substitution keeps `X'WX + λS` factorisable and
    /// preserves β̂, but it makes `A` not the Hessian of the penalised
    /// deviance on the outlier rows, and the IFT bracket must be that
    /// Hessian. Same `Dmu2` expression as `irls_observed_pair`, so the two
    /// agree exactly on core rows.
    fn observed_curvature_weights(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<ndarray::Array1<f64>> {
        use ndarray::Array1;
        let n = y.len();
        let nu = self.nu;
        let q = nu * self.sigma2;
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            // Identity link → μ = η, dμ/dη = 1, so the η-coordinate weight
            // collapses to ½·Dmu2 (mgcv `gam.fit4.r:367`: `w = dd$Deta2 * .5`).
            let r = y[i] - eta[i];
            let s = q + r * r;
            let dmu2 = 2.0 * (nu + 1.0) * (q - r * r) / (s * s);
            let pw = prior_w.map(|p| p[i]).unwrap_or(1.0);
            w[i] = pw * 0.5 * dmu2;
        }
        Some(w)
    }

    /// Trace-term weight derivatives consistent with `irls_observed_pair`'s
    /// observed/expected weight switch (see `Loss::ift_trace_weight_derivs`).
    ///
    /// `A` (hence the score's `log|H|`) uses `W = ½·d²D/dμ²` on core rows and
    /// the expected curvature `W = ½·EDmu2 = (ν+1)/((ν+3)σ²)` on outlier rows
    /// (`|r| ≥ √(νσ²)`). We return `∂W/∂θ`, `∂W/∂μ` of that same `W`:
    ///   - **core**: `∂W/∂θ_k = ½·dmu2th[k]`, `∂W/∂μ = ½·dmu3`.
    ///   - **outlier**: `W = (ν+1)/((ν+3)σ²)` is μ-independent, so `∂W/∂μ = 0`
    ///     and (matching mgcv `Dd`'s `EDmu2th`, `efam.r:1344`, with the ½):
    ///       `∂W/∂(log σ²)     = −W`,
    ///       `∂W/∂(log(ν − 2)) = 2(ν−2)/((ν+3)²σ²)`.
    fn ift_trace_weight_derivs(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<(ndarray::Array2<f64>, ndarray::Array1<f64>)> {
        use ndarray::{Array1, Array2};
        let n = y.len();
        let nu = self.nu;
        let sigma2 = self.sigma2;
        let nu_p1 = nu + 1.0;
        let nu_minus_df = nu - MIN_DF;
        let q = nu * sigma2; // νσ²
        let qs_theta1 = sigma2 * nu_minus_df; // ∂q/∂θ_1
                                              // Expected weight W_exp = ½·EDmu2 and its θ-derivatives.

        let mut dw_dtheta = Array2::<f64>::zeros((n, 2));
        let mut dw_dmu = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = y[i] - eta[i];
            let r2 = r * r;
            let s = q + r2;
            let s2 = s * s;
            let s3 = s2 * s;
            let wt_i = prior_w.map(|w| w[i]).unwrap_or(1.0);

            // `∂W/∂θ_k = ½·dmu2th[k]`, `∂W/∂μ = ½·dmu3` (Level-1 formulas), on
            // EVERY row. No outlier branch: this hook's contract is
            // "derivatives of the weight actually in the matrix the score
            // differentiates", and that matrix now carries the observed
            // curvature everywhere (`pirls.rs` rebuilds `a_factor` from
            // `observed_curvature_weights`). There used to be an
            // expected-curvature branch here mirroring `irls_observed_pair`'s
            // substitution; it went with the substitution.
            let dmu2th_0 = 2.0 * nu_p1 * q * (3.0 * r2 - q) / s3;
            let dmu2th_1 =
                2.0 * (nu_minus_df * (q - r2) * s + nu_p1 * qs_theta1 * (3.0 * r2 - q)) / s3;
            let dmu3 = 4.0 * r * nu_p1 * (3.0 * q - r2) / s3;
            dw_dtheta[[i, 0]] = wt_i * 0.5 * dmu2th_0;
            dw_dtheta[[i, 1]] = wt_i * 0.5 * dmu2th_1;
            dw_dmu[i] = wt_i * 0.5 * dmu3;
        }
        Some((dw_dtheta, dw_dmu))
    }

    /// Per-row Level-2 derivatives feeding the full analytic Hessian path
    /// (port of mgcv_rust `src/reml/mod.rs::tdist_dd_arrays` lines
    /// 1267–1329's Level-2 outputs: `det4`, `det3_th`, `dth2`, `det_th2`,
    /// `det2_th2`).
    ///
    /// Returned in gamrs's outer-Newton convention `θ = [log σ², log(ν−2)]`.
    /// mgcv_rust's `tdist_dd_arrays` works in `[log(ν−2), log σ]`; the
    /// `log σ → log σ²` Jacobian is `d/d(log σ²) = (1/2)·d/d(log σ)` and
    /// is applied here per axis (×½ on the σ² axis, ×¼ on the σ²×σ²
    /// pair). The (ν−2) axis is identical in both conventions.
    ///
    /// Cross-reference for the assembly: `tdist_gdi2_native` at
    /// `nn_exploring/src/reml/mod.rs:1338-1564`. The outer-vs-native
    /// remap function `reml_joint_gh_gamfit4_tdist_analytic` at
    /// `nn_exploring/src/reml/mod.rs:1687-1745` documents the same
    /// ½ / ¼ Jacobian.
    fn level2_shape_derivatives(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<crate::traits::Level2ShapeDerivs> {
        use ndarray::{Array1, Array2};
        let n = y.len();
        let nu = self.nu;
        let sigma2 = self.sigma2;
        let nu_p1 = nu + 1.0;
        let nu_minus_df = nu - MIN_DF;
        let q = nu * sigma2;
        // Mgcv intermediates from `tdist_dd_arrays` lines 1218-1264.
        // We re-derive locally for clarity / inlining vs calling a
        // separate helper.
        let nu1nu = nu_p1 / nu;
        let nu2nu = nu_minus_df / nu;

        let mut dmu4 = Array1::<f64>::zeros(n);
        // (n × 2)  — outer order [log σ², log(ν − 2)]
        let mut dmu3_th = Array2::<f64>::zeros((n, 2));
        // packed (0,0)→0, (0,1)→1, (1,1)→2
        let mut dth2 = Array2::<f64>::zeros((n, 3));
        let mut dmu_th2 = Array2::<f64>::zeros((n, 3));
        let mut dmu2_th2 = Array2::<f64>::zeros((n, 3));

        for i in 0..n {
            let r = y[i] - eta[i];
            let r2 = r * r;
            let s = q + r2;
            let s2 = s * s;
            let _s3 = s2 * s;
            let wt = prior_w.map(|w| w[i]).unwrap_or(1.0);

            // Mgcv's per-row intermediates (lines 1253-1264). All scoped
            // to this iteration so the loop fits in registers.
            let a = 1.0 + r2 / q; // = s / q
            let sig2a = sigma2 * a; // = s / nu
            let nusig2a = s; // = ν σ² + r²
            let f = nu_p1 * r / nusig2a; // (ν+1) r / s
            let f1 = r / nusig2a; // r / s
            let nu1nusig2a = nu_p1 / nusig2a; // (ν+1) / s
            let fym = f * r; // (ν+1) r² / s
            let ff1 = f * f1; // (ν+1) r² / s²
            let f1ym = f1 * r; // r² / s
            let fymf1 = fym * f1; // (ν+1) r³ / s²
            let ymsig2a = r / sig2a; // ν r / s
            let fymf1ym = fym * f1ym; // (ν+1) r⁴ / s²
            let f1ymf1 = f1ym * f1; // r³ / s²  (mgcv R `f1ymf1`)
                                    // NB: mgcv R `efam.r:1373-1375`'s Dmu2th2[,2]/[,3] uses `f1ymf1`
                                    // (= f1ym·f1 = r³/s²), NOT `fymf1` (= fym·f1 = (ν+1)·r³/s²).
                                    // mgcv_rust's `tdist_dd_arrays` mis-copies the symbol from
                                    // `Dmuth2` (which DOES use `fymf1`) into Dmu2th2 — verified
                                    // by FD at the gamrs Level-2 boundary tests. We use the R
                                    // formula directly here.

            // ── ∂⁴D / ∂μ⁴  (mgcv `det4`, line 1270) ─────────────────────
            dmu4[i] =
                wt * 12.0 * (-nu1nusig2a / nusig2a + 8.0 * ff1 / nusig2a - 8.0 * ff1 * f1 * f1);

            // ── ∂⁴D / (∂μ³ ∂θ_k)  in NATIVE order [log(ν−2), log σ] ────
            //   mgcv `det3_th[0]` = ∂(∂³D/∂μ³)/∂(log(ν−2)),
            //   mgcv `det3_th[1]` = ∂(∂³D/∂μ³)/∂(log σ).
            let det3_th_nu = wt
                * 4.0
                * (-6.0 * f / nusig2a + 3.0 * f1 / sig2a + 18.0 * ff1 * f1
                    - 4.0 * f1ymf1 / sig2a
                    - 12.0 * nu_p1 * r * f1.powi(4))
                * nu2nu;
            let det3_th_logsigma =
                wt * 48.0 * f * (-1.0 / nusig2a + 3.0 * f1 * f1 - 2.0 * f1ymf1 * f1);
            //   gamrs outer order [log σ², log(ν−2)]:
            //     col 0 = log σ² ← native log σ × (1/2 Jacobian)
            //     col 1 = log(ν−2) ← native log(ν−2) (identity)
            dmu3_th[[i, 0]] = 0.5 * det3_th_logsigma;
            dmu3_th[[i, 1]] = det3_th_nu;

            // ── ∂²D / (∂θ_i ∂θ_j)  (mgcv `dth2`, lines 1282-1287) ──────
            //   mgcv native packing: (0,0)=νν, (0,1)=νσ, (1,1)=σσ.
            let dth2_nu_nu = wt
                * (nu_minus_df * a.ln()
                    + nu2nu
                        * r
                        * r
                        * (-2.0 * nu_minus_df - nu_p1 + 2.0 * nu_p1 * nu2nu
                            - nu_p1 * nu2nu * f1ym)
                        / nusig2a);
            let dth2_nu_logsigma = wt * 2.0 * (fym - r * ymsig2a - fymf1ym) * nu2nu;
            let dth2_logsigma_logsigma = wt * 4.0 * fym * (1.0 - f1ym);
            //   gamrs outer packing: (0,0)=σ²σ², (0,1)=σ²ν, (1,1)=νν.
            //     σ²σ² ← σσ × ¼   (two Jacobian factors)
            //     σ²ν ← σν × ½    (one Jacobian factor)
            //     νν  ← νν        (identity)
            dth2[[i, 0]] = 0.25 * dth2_logsigma_logsigma;
            dth2[[i, 1]] = 0.5 * dth2_nu_logsigma;
            dth2[[i, 2]] = dth2_nu_nu;

            // ── ∂³D / (∂μ ∂θ_i ∂θ_j)  (mgcv `det_th2`, lines 1290-1299) ─
            let term = 2.0 * nu2nu - 2.0 * nu1nu * nu2nu - 1.0 + nu1nu;
            let det_th2_nu_nu = wt
                * 2.0
                * f1
                * nu_minus_df
                * (term - 2.0 * nu2nu * f1ym + 4.0 * fym * nu2nu / nu
                    - fym / nu
                    - 2.0 * fymf1ym * nu2nu / nu);
            let det_th2_nu_logsigma = wt
                * 4.0
                * (-f + ymsig2a + 3.0 * fymf1 - ymsig2a * f1ym - 2.0 * fymf1 * f1ym)
                * nu2nu;
            let det_th2_logsigma_logsigma = wt * 8.0 * f * (-1.0 + 3.0 * f1ym - 2.0 * f1ym * f1ym);
            dmu_th2[[i, 0]] = 0.25 * det_th2_logsigma_logsigma;
            dmu_th2[[i, 1]] = 0.5 * det_th2_nu_logsigma;
            dmu_th2[[i, 2]] = det_th2_nu_nu;

            // ── ∂⁴D / (∂μ² ∂θ_i ∂θ_j)  (mgcv `det2_th2`, lines 1307-1328)
            let det2_th2_nu_nu = wt
                * 2.0
                * nu_minus_df
                * (-term + 10.0 * nu2nu * f1ym - 16.0 * fym * nu2nu / nu - 2.0 * f1ym
                    + 5.0 * nu1nu * f1ym
                    - 8.0 * nu2nu * f1ym * f1ym
                    + 26.0 * fymf1ym * nu2nu / nu
                    - 4.0 * nu1nu * f1ym * f1ym
                    - 12.0 * nu1nu * nu2nu * f1ym * f1ym * f1ym)
                / nusig2a;
            let det2_th2_nu_logsigma = wt
                * 4.0
                * (nu1nusig2a - 1.0 / sig2a - 11.0 * nu_p1 * f1 * f1
                    + 5.0 * f1ym / sig2a
                    + 22.0 * nu_p1 * f1ymf1 * f1
                    - 4.0 * f1ym * f1ym / sig2a
                    - 12.0 * nu_p1 * f1ymf1 * f1ymf1)
                * nu2nu;
            let det2_th2_logsigma_logsigma = wt
                * 8.0
                * (nu1nusig2a - 11.0 * nu_p1 * f1 * f1 + 22.0 * nu_p1 * f1ymf1 * f1
                    - 12.0 * nu_p1 * f1ymf1 * f1ymf1);
            dmu2_th2[[i, 0]] = 0.25 * det2_th2_logsigma_logsigma;
            dmu2_th2[[i, 1]] = 0.5 * det2_th2_nu_logsigma;
            dmu2_th2[[i, 2]] = det2_th2_nu_nu;
        }

        Some(crate::traits::Level2ShapeDerivs {
            dmu4,
            dmu3_th,
            dth2,
            dmu_th2,
            dmu2_th2,
        })
    }

    /// `Σᵢ wt_i · ∂²ls(y_i)/(∂θ_i ∂θ_j)` for the two scat shape axes
    /// `θ = [log σ², log(ν − 2)]`. Packed upper-triangular per
    /// `shape_pair_index`: `[0]=σ²σ², [1]=σ²ν, [2]=νν`.
    ///
    /// Per-obs ls is `log Γ((ν+1)/2) − log Γ(ν/2) − 0.5·log(π·ν·σ²)`.
    /// Direct computation in gamrs's outer convention:
    /// - `∂²ls/∂(log σ²)² = 0` (the `−0.5·log σ²` is linear in `log σ²`).
    /// - `∂²ls/(∂(log σ²) ∂(log(ν − 2))) = 0` (factorable).
    /// - `∂²ls/∂(log(ν − 2))² = (ν − 2)/2 · {(ν − 2)/2 · [ψ'((ν+1)/2)
    ///       − ψ'(ν/2)] + [ψ((ν+1)/2) − ψ(ν/2)] + ((ν − 2)/ν − 1)/ν}`.
    ///
    /// The mgcv_rust counterpart is `ls2[0,0]` in `tdist_gdi2_native`
    /// (line 1428-1432); the σ row/col are 0 there too (matches our
    /// `[0]=0`, `[1]=0`) but the indexing differs — mgcv's index 0 is
    /// log(ν−2) so its `ls2[0,0]` maps to gamrs's pair `[2]=νν`.
    fn sum_saturated_log_lik_d2theta(
        &self,
        y: ndarray::ArrayView1<f64>,
        _scale: f64,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Vec<f64> {
        use crate::special::{digamma, trigamma};
        let nu = self.nu;
        let nu_minus_df = nu - MIN_DF;
        let half_nu_p1 = (nu + 1.0) / 2.0;
        let half_nu = nu / 2.0;
        let nu2nu = nu_minus_df / nu;
        let sum_w: f64 = match prior_w {
            Some(w) => w.iter().sum(),
            None => y.len() as f64,
        };
        // ∂²ls/∂(log(ν−2))² (derived above; matches mgcv_rust line 1428-1432
        // with the substitution `nu2 = ν − 2`, `nu2nu = (ν − 2)/ν`).
        let d2ls_dnu2 =
            nu_minus_df * nu_minus_df * 0.25 * (trigamma(half_nu_p1) - trigamma(half_nu))
                + nu_minus_df * 0.5 * (digamma(half_nu_p1) - digamma(half_nu))
                + 0.5 * nu2nu * nu2nu
                - 0.5 * nu2nu;
        vec![0.0, 0.0, sum_w * d2ls_dnu2]
    }
}

impl VarianceFn for TVariance {
    /// Constant variance σ² (NOT μ-dependent — t is location-scale).
    fn variance(&self, _mu: f64) -> f64 {
        self.sigma2
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        // Sync `sigma2` from the first transformed param (log σ²). Ignores
        // the ν transform — variance doesn't depend on ν for scat.
        debug_assert_eq!(
            params.len(),
            2,
            "TVariance expects 2 shape params (slot 0 is log σ²)"
        );
        self.sigma2 = params[0].exp();
    }
}

/// Phase 2a convenience constructor — TDist + identity link at given shape.
pub fn tdist_identity(nu: f64, sigma2: f64) -> Family<TDist, IdentityLink, TVariance> {
    Family::new(TDist { nu, sigma2 }, IdentityLink, TVariance { sigma2 })
}
