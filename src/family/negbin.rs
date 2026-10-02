//! Negative Binomial with log link.

use crate::traits::{Level1ShapeDerivs, Loss, VarianceFn};

use super::link::LogLink;
use super::Family;

/// Negative binomial likelihood for over-dispersed counts. mgcv's `nb()`:
/// `V(μ) = μ + μ²/θ` (Poisson-like + quadratic over-dispersion term).
/// Canonical link is `log(μ/(μ+θ))` but mgcv uses `log` by convention
/// (non-canonical but standard).
///
/// One shape parameter `θ > 0` (over-dispersion). Transform: `log θ`.
/// θ small → heavy over-dispersion (variance dominated by μ²/θ); θ → ∞
/// recovers Poisson.
#[derive(Clone)]
pub struct NegBin {
    pub theta: f64,
}

/// μ-dependent variance for NegBin: `V(μ) = μ + μ²/θ`. `θ` must be kept in
/// sync with the Loss via `set_shape_params`.
#[derive(Clone)]
pub struct NegBinVariance {
    pub theta: f64,
}

impl Loss for NegBin {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        // Same `max(y, 0.1)` as Poisson — keeps log-link domain valid.
        y.iter().map(|&yi| yi.max(0.1)).collect()
    }

    /// `D(y, μ) = 2[y·log(max(1,y)/μ) - (y+θ)·log((y+θ)/(μ+θ))]`. mgcv
    /// `negbin$dev.resids`. For y=0: `D = 2θ·log((μ+θ)/θ)`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let theta = self.theta;
        let mu = mu.max(1e-300);
        if y > 0.0 {
            2.0 * (y * (y / mu).ln() - (y + theta) * ((y + theta) / (mu + theta)).ln())
        } else {
            2.0 * theta * ((mu + theta) / theta).ln()
        }
    }

    /// Saturated log-lik at μ=y: `lgamma(y+θ) - lgamma(θ) - lgamma(y+1)
    /// + y·log(y/(y+θ)) + θ·log(θ/(y+θ))`. All four terms are kept —
    /// matches mgcv `gam.fit3.r:2497-2548` (`fix.family.ls`) byte-for-byte
    /// (v0.x `src/pirls/mod.rs::Family::NegBin` lines 565-581). The
    /// `lgamma(y+1)` piece is constant in θ AND λ, so it doesn't affect
    /// the score's optimum — but its inclusion makes the absolute REML
    /// value commensurable with mgcv's reported `score`, which lets the
    /// parity diagnostic compare component-by-component to ≤ 1e-12 instead
    /// of off-by-`Σ lgamma(y+1)`. Closes the `ls` parity gap on the
    /// 2026-05-28 NegBin layer-4 cross-eval (component diff went from
    /// ~737 to 0). φ is fixed at 1 (NegBin's dispersion lives entirely in
    /// θ), so `_scale` is moot.
    fn saturated_log_lik(&self, y: f64, _scale: f64) -> f64 {
        let theta = self.theta;
        let yt = y + theta;
        let lg = crate::special::log_gamma(yt)
            - crate::special::log_gamma(theta)
            - crate::special::log_gamma(y + 1.0);
        let y_term = if y > 0.0 { y * (y / yt).ln() } else { 0.0 };
        let t_term = theta * (theta / yt).ln();
        lg + y_term + t_term
    }

    /// `∂D/∂μ = 2θ(μ - y) / [μ(μ + θ)]`. Unified across y=0 and y>0.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let theta = self.theta;
        let mu = mu.max(1e-300);
        2.0 * theta * (mu - y) / (mu * (mu + theta))
    }

    /// `∂²D/∂μ² = 2θ · [-μ² + 2yμ + yθ] / [μ²(μ + θ)²]`.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let theta = self.theta;
        let mu = mu.max(1e-300);
        let num = -mu * mu + 2.0 * y * mu + y * theta;
        let denom = mu * mu * (mu + theta) * (mu + theta);
        2.0 * theta * num / denom
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0) // σ² fixed; θ is the shape param, not φ.
    }

    fn n_shape_params(&self) -> usize {
        1
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(params.len(), 1, "NegBin expects 1 shape param (log θ)");
        self.theta = params[0].exp();
    }
    fn get_shape_params(&self) -> Vec<f64> {
        vec![self.theta.ln()]
    }

    /// Non-canonical log link → use Newton observed-info IRLS weight, not
    /// Fisher. mgcv's `nb()` runs through gam.fit5 which is observed-info
    /// at the score level; gamrs's PIRLS Newton-IRLS path is the same
    /// `wf·α = ½·Dmu2(η)` quantity (verified by NegBin derivation, see
    /// docs/level1_shape_derivs_conventions.md). Opting in makes the
    /// trait convention `Level1ShapeDerivs::dmu3 = ∂³D/∂μ³` consistent with
    /// the inner-solver W at score/shape_aware/gradient.rs:147.
    fn use_newton_irls(&self) -> bool {
        true
    }

    /// NegBin's W = 1/(V·g'²) = 1/(μ + μ²/θ) for log link — stable under
    /// small β perturbations, so eligible for the mgcv_rust NoRefresh IFT
    /// line-search shortcut. (Skip list excludes TDist, Quantile,
    /// InverseGaussian, Tweedie; NegBin is the headline beneficiary.)
    fn allows_no_refresh(&self) -> bool {
        true
    }

    /// NegBin's analytic-Hessian profile-θ Newton converges cleanly to
    /// loose tolerances. Sweep (`scripts/sweep_tolerances.py`): 5e-6/1e-6
    /// shows zero ρ̂ drift vs 1e-9 baseline + 24% wall-time win on 1D NB
    /// (n=300), and no impact on 2D NB (n=600). The mgcv default
    /// (5e-7/1e-7) over-converges here.
    fn outer_tuning(&self) -> crate::outer::OuterTuning {
        crate::outer::OuterTuning {
            grad_tol: 5.0e-6,
            reml_tol: 1.0e-6,
            ..crate::outer::OuterTuning::mgcv_default()
        }
    }

    /// Loosened from mgcv's `1e-8` (`src/lib.rs:1277`) to `1e-6` because
    /// NegBin's PIRLS converges slowly (V = μ + μ²/θ is more complex
    /// than μ or μ²). At 1e-8 we average ~10.6 IRLS iters/call on the
    /// 1D NB fixture; at 1e-6 we get ~6 iters/call with no measurable
    /// parity hit (μ rel-err vs mgcv R unchanged at 2.42e-3, ρ̂
    /// agreement to 1e-5).
    fn pirls_dev_rel_tol(&self) -> f64 {
        1.0e-6
    }

    /// Per-row Level-1 NegBin derivatives `(Dmu3, Dth, Dmuth, Dmu2th)` for
    /// the shape-aware envelope score's Tk·KK' β-chain term in the
    /// ρ-gradient (`src/score/shape_aware/gradient.rs:147`) and for the
    /// analytic shape (`log θ`) gradient via the IFT path.
    ///
    /// Convention `dmu3 := ∂³D/∂μ³` per the `Level1ShapeDerivs` doc; the
    /// consumer mathematically requires `½·dmu3 = ∂W/∂η`, which holds
    /// because gamrs opts NegBin into Newton-IRLS weights `W = ½·Dmu2(η)`
    /// via `use_newton_irls()` above. See
    /// `docs/level1_shape_derivs_conventions.md` for the full derivation.
    ///
    /// Derivatives from `Dmu(μ) = -2y/μ + 2(θ+y)/(μ+θ)`:
    /// - **Dmu3** = `∂³D/∂μ³` = `-4y/μ³ + 4(θ+y)/(μ+θ)³`.
    /// - **Dth** = `θ · ∂D/∂θ` = `-2θ·log((y+θ)/(μ+θ)) - 2θ(μ-y)/(μ+θ)`.
    /// - **Dmuth** = `θ · ∂Dmu/∂θ` = `2θ(μ-y)/(μ+θ)²`.
    /// - **Dmu2th** = `θ · ∂Dmu²/∂θ` = `2θ(2y+θ-μ)/(μ+θ)³`.
    ///
    /// `log θ` transform: `∂/∂(log θ) = θ · ∂/∂θ` — all θ-derivatives carry
    /// the leading `θ` factor. Prior weights are baked in (mirrors ocat).
    /// `eta` is converted to `μ = exp(η)` (log link).
    fn level1_shape_derivatives(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<Level1ShapeDerivs> {
        Some(negbin_dd_level1(y, eta, self.theta, prior_w))
    }

    /// `Σᵢ ∂ls_i/∂(log θ)` — closed form for NegBin. The saturated log-lik
    /// `ls(y; θ) = lgamma(y+θ) - lgamma(θ) - lgamma(y+1) + y·log(y/(y+θ))
    /// + θ·log(θ/(y+θ))` has θ-derivative (via lgamma' = digamma):
    /// `∂ls/∂θ = ψ(y+θ) - ψ(θ) + log(θ/(y+θ))`. With `α = log θ` the chain
    /// factor is `θ`, so `∂ls/∂α = θ·[ψ(y+θ) - ψ(θ) + log(θ/(y+θ))]`. The
    /// IFT shape-gradient consumer at `gradient.rs:343` subtracts the
    /// returned sum to close the `gam.fit5.r:1668 -ls$d1` row — without
    /// this, NegBin's IFT shape gradient ships the wrong sign because ocat
    /// (the original Level1 client) has `ls ≡ 0` and the consumer skipped
    /// the term.
    fn sum_saturated_log_lik_dtheta(
        &self,
        y: ndarray::ArrayView1<f64>,
        _scale: f64,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Vec<f64> {
        use crate::special::digamma;
        let theta = self.theta;
        let psi_theta = digamma(theta);
        let mut acc = 0.0_f64;
        for i in 0..y.len() {
            let yi = y[i];
            let wt = prior_w.map(|w| w[i]).unwrap_or(1.0);
            let yt = yi + theta;
            // ∂ls / ∂(log θ) = θ · [ψ(y+θ) - ψ(θ) + log(θ/(y+θ))]
            let term = theta * (digamma(yt) - psi_theta + (theta / yt).ln());
            acc += wt * term;
        }
        vec![acc]
    }
}

/// Per-row Level-1 NegBin derivatives at the converged η for the current
/// θ — port-spirit of `ocat::ocat_dd_level1` for the single-shape-param
/// NegBin case. Returns `(Dmu3, Dth, Dmuth, Dmu2th)` in the
/// `Level1ShapeDerivs` layout: `Dmu3` is length-`n`, the other three are
/// `(n × 1)` (NegBin has one shape param `α = log θ`).
///
/// `eta` is on the linear-predictor scale; we map to `μ = exp(η)` (log
/// link). Prior weights are multiplied into every row (mgcv convention,
/// matching ocat's level-1 impl). `μ` is floored at `1e-300` to keep the
/// closed-form ratios finite at the saturation boundary.
pub fn negbin_dd_level1(
    y: ndarray::ArrayView1<f64>,
    eta: ndarray::ArrayView1<f64>,
    theta: f64,
    prior_w: Option<ndarray::ArrayView1<f64>>,
) -> Level1ShapeDerivs {
    use ndarray::{Array1, Array2};
    let n = y.len();
    debug_assert_eq!(eta.len(), n);

    let mut dmu3 = Array1::<f64>::zeros(n);
    let mut dth = Array2::<f64>::zeros((n, 1));
    let mut dmuth = Array2::<f64>::zeros((n, 1));
    let mut dmu2th = Array2::<f64>::zeros((n, 1));

    for i in 0..n {
        let wt_i = prior_w.map(|w| w[i]).unwrap_or(1.0);
        let yi = y[i];
        let mu_i = eta[i].exp().max(1e-300);
        let mut_t = mu_i + theta;

        // ∂³D/∂μ³ = -4y/μ³ + 4(θ+y)/(μ+θ)³.
        let mu3 = mu_i * mu_i * mu_i;
        let mut_t3 = mut_t * mut_t * mut_t;
        dmu3[i] = wt_i * (-4.0 * yi / mu3 + 4.0 * (theta + yi) / mut_t3);

        // ∂D/∂(log θ): branch on y=0 only to avoid log(0).
        let log_ratio = if yi > 0.0 {
            ((yi + theta) / mut_t).ln()
        } else {
            (theta / mut_t).ln()
        };
        dth[[i, 0]] = wt_i * theta * (-2.0 * log_ratio - 2.0 * (mu_i - yi) / mut_t);

        // ∂Dmu/∂(log θ) = 2θ(μ - y)/(μ+θ)².
        let mut_t2 = mut_t * mut_t;
        dmuth[[i, 0]] = wt_i * theta * 2.0 * (mu_i - yi) / mut_t2;

        // ∂Dmu²/∂(log θ) = 2θ(2y + θ - μ)/(μ+θ)³.
        dmu2th[[i, 0]] = wt_i * theta * 2.0 * (2.0 * yi + theta - mu_i) / mut_t3;
    }

    Level1ShapeDerivs {
        dmu3,
        dth,
        dmuth,
        dmu2th,
    }
}

impl VarianceFn for NegBinVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu + mu * mu / self.theta
    }
    fn d_variance(&self, mu: f64) -> f64 {
        1.0 + 2.0 * mu / self.theta
    }
    fn d2_variance(&self, _mu: f64) -> f64 {
        2.0 / self.theta
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(
            params.len(),
            1,
            "NegBinVariance expects 1 shape param (log θ)"
        );
        self.theta = params[0].exp();
    }
}

/// Phase 6 convenience constructor — NegBin + log link at given θ₀.
pub fn negbin_log(theta: f64) -> Family<NegBin, LogLink, NegBinVariance> {
    Family::new(NegBin { theta }, LogLink, NegBinVariance { theta })
}
