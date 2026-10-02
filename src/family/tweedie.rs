//! Tweedie with log link (compound-Poisson-Gamma for `1 < p < 2`).

use crate::traits::{Loss, VarianceFn};

use super::link::LogLink;
use super::Family;

/// Tweedie compound-Poisson-Gamma likelihood for `1 < p < 2`. Handles
/// zero-inflated continuous data (insurance claims, rainfall) by mixing
/// a point mass at 0 (Poisson-like) with continuous positive responses
/// (Gamma-like).
///
/// - Deviance (1 < p < 2): `D(y, μ) = 2·[y^(2-p)/((1-p)(2-p)) - y·μ^(1-p)/(1-p) + μ^(2-p)/(2-p)]` for y > 0; `D = 2·μ^(2-p)/(2-p)` for y = 0.
/// - Variance: `V(μ) = μ^p`.
/// - **Two shape params**: `[log φ, p_transform]` where
///   `p_transform = log((p - 1)/(2 - p))` so `p ∈ (1, 2)` via
///   `p = 1 + 1/(1 + exp(-θ_p))`. Joint outer Newton on `[log λ, log φ, p_transform]`.
/// - Saturated log-lik: Dunn-Smyth series sum (`crate::special::tweedie_log_w`).
#[derive(Clone)]
pub struct Tweedie {
    /// Variance power, must be in (1, 2).
    pub p: f64,
    /// Dispersion parameter.
    pub phi: f64,
    /// When `true` (mgcv `tw()`): `p` is profiled — there are 2 shape
    /// params `[log φ, p_transform]` and the outer Newton estimates `p`
    /// jointly. When `false` (mgcv `Tweedie(p=val)`): `p` is held CONSTANT
    /// at the constructed value — there is 1 shape param `[log φ]` and the
    /// p-axis is dropped from every shape derivative so Newton optimizes φ
    /// (+ λ) only. See `tweedie_log` (profile) vs `tweedie_log_fixed_p`.
    pub profile_p: bool,
}

#[derive(Clone)]
pub struct TweedieVariance {
    pub p: f64,
    /// Mirror of `Tweedie::profile_p` so `set_shape_params` knows whether
    /// the incoming slice carries a `p_transform` entry to consume.
    pub profile_p: bool,
}

impl Loss for Tweedie {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        // Tweedie μ > 0. Same floor as Gamma; doesn't need to be small
        // because y=0 is allowed in Tweedie (compound Poisson mass at 0).
        y.iter().map(|&yi| yi.max(0.1)).collect()
    }

    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let p = self.p;
        let mu = mu.max(1e-300);
        let twop = 2.0 - p;
        let onep = 1.0 - p;
        if y > 0.0 {
            2.0 * (y.powf(twop) / (onep * twop) - y * mu.powf(onep) / onep + mu.powf(twop) / twop)
        } else {
            2.0 * mu.powf(twop) / twop
        }
    }

    /// Tweedie saturated log-lik at μ=y:
    /// `ls(y; φ, p) = l_base(y; φ, p) - log(y) + log W(y; φ, p)` for y > 0,
    /// 0 for y = 0, where
    /// `l_base = y^(2-p) / ((1-p)(2-p)·φ)`. Ported from v0.x
    /// `src/pirls/mod.rs::saturated_log_likelihood` Tweedie branch.
    ///
    /// Previously gamrs dropped the `l_base` and `-log(y)` pieces (the
    /// "Conservative" path). `-log(y)` is y-only and constant in (φ, p),
    /// so it doesn't shift the gradient, but `l_base ∝ 1/φ` is
    /// load-bearing for the log-φ score gradient — its omission was the
    /// root cause of the wrong-minimum trap (Phase-1 v0.2 port,
    /// 2026-05-24).
    ///
    /// Tweedie owns its φ via `self.phi`; the score body wires Tweedie
    /// through `OwnedByLossProfile` which passes `self.phi` as the
    /// `scale` argument. So `scale == self.phi` BY CONSTRUCTION at every
    /// caller — the debug assertion below catches any future drift if a
    /// new caller wires a different Profile.
    fn saturated_log_lik(&self, y: f64, scale: f64) -> f64 {
        debug_assert!(
            (scale - self.phi).abs() < 1e-12 * (self.phi.abs().max(1.0)),
            "Tweedie::saturated_log_lik received scale={scale} ≠ self.phi={}; \
             this family must be wired through OwnedByLossProfile so the \
             passed scale tracks the shape-managed φ.",
            self.phi
        );
        if y <= 0.0 {
            return 0.0;
        }
        let p = self.p;
        let phi = self.phi.max(1e-12);
        let onep = 1.0 - p; // < 0 for 1 < p < 2
        let twop = 2.0 - p; // > 0
                            // l_base = y^(2-p) / ((1-p)(2-p)·φ)
        let l_base = y.powf(twop) / (onep * twop * phi);
        let log_w = crate::special::tweedie_log_w(y, phi, p);
        l_base - y.ln() + log_w
    }

    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let p = self.p;
        let mu = mu.max(1e-300);
        2.0 * (mu - y) / mu.powf(p)
    }

    /// The β-chain of `log|H|` in the ρ-gradient. Tweedie's `A` carries the
    /// Fisher weight `W = μ^(2−p)` (log link, `1/(V·g'²)`), so along η
    /// `∂W/∂η = (2 − p)·W`. Without this the ρ-gradient was the bare envelope
    /// form, off by 0.024 on a two-smooth fit — enough for the outer Newton to
    /// stop where the true gradient was not zero. The θ-columns are unused on
    /// Tweedie's analytic shape-gradient path.
    fn ift_trace_weight_derivs(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<(ndarray::Array2<f64>, ndarray::Array1<f64>)> {
        // The Newton (observed) weight `log|H|` is taken off — mgcv's `tw()` is an
        // extended family and uses it:
        //   W = μ^(2−p) + (p−1)(y−μ)μ^(1−p)
        //   ∂W/∂η = (2−p)μ^(2−p) + (p−1)μ^(1−p)·[(1−p)(y−μ) − μ]
        //   ∂W/∂p = −ln μ·W + (y−μ)μ^(1−p)        (fixed μ; log φ does not enter W)
        let n = y.len();
        let p = self.p;
        let dp_dtheta = tweedie_dp_dtheta(p);
        let mut dw_dtheta = ndarray::Array2::<f64>::zeros((n, self.n_shape_params()));
        let dw_deta = ndarray::Array1::from_shape_fn(n, |i| {
            let pw = prior_w.map(|w| w[i]).unwrap_or(1.0);
            let mu = eta[i].exp();
            let m1 = ((1.0 - p) * eta[i]).exp(); // μ^(1−p)
            let m2 = mu * m1; // μ^(2−p)
            let r = y[i] - mu;
            let w = m2 + (p - 1.0) * r * m1;
            if self.profile_p {
                dw_dtheta[[i, 1]] = pw * (-eta[i] * w + r * m1) * dp_dtheta;
            }
            pw * ((2.0 - p) * m2 + (p - 1.0) * m1 * ((1.0 - p) * r - mu))
        });
        Some((dw_dtheta, dw_deta))
    }

    fn shape_score_residual_derivs(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<ndarray::Array2<f64>> {
        // r = w·(y − μ)·μ^(1−p) (log link); ∂r/∂p = −r·ln μ. β̂ does not depend on log φ.
        let n = y.len();
        let mut out = ndarray::Array2::<f64>::zeros((n, self.n_shape_params()));
        if self.profile_p {
            let dp_dtheta = tweedie_dp_dtheta(self.p);
            for i in 0..n {
                let mu = eta[i].exp();
                let r = prior_w.map(|w| w[i]).unwrap_or(1.0)
                    * (y[i] - mu)
                    * ((1.0 - self.p) * eta[i]).exp();
                out[[i, 1]] = -r * eta[i] * dp_dtheta;
            }
        }
        Some(out)
    }

    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        // ∂/∂μ [2(μ - y) · μ^(-p)] = 2μ^(-p) · [1 - p(μ-y)/μ]
        // = 2·[μ - p(μ-y)] / μ^(p+1) = 2·[(1-p)μ + p·y] / μ^(p+1)
        let p = self.p;
        let mu = mu.max(1e-300);
        2.0 * ((1.0 - p) * mu + p * y) / mu.powf(p + 1.0)
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        // Tweedie φ IS shape-managed (via shape params) but the score
        // dispatch uses the live `self.phi` as a "fixed" dispersion at
        // each outer probe. The shape-aware score updates `self.phi`
        // before each PIRLS call.
        Some(self.phi)
    }

    // NOTE on `score_rank_adjustment`: Tweedie INTENTIONALLY keeps the
    // default `0` rank convention, NOT the `-1` ocat uses. Tweedie multi-
    // smooth parity diagnostic 2026-05-28 confirmed:
    //
    //   * With rank_adj=0 (default): gamrs and v0.x converge to the SAME
    //     `(λ, p)` and predictions agree to `μ-RMSE = 0.0023%` (essentially
    //     identical). Per-component `log_det_lambda_s` reads `~17` units
    //     higher in gamrs but `log_det_h` reads `~9` units lower (v0.x's
    //     larger score-side ridge inflates log|H|), and the two
    //     differences cancel through the outer Newton's stationarity.
    //   * With rank_adj=-1: the components-level `log_det_lambda_s`
    //     matches v0.x byte-for-byte BUT gamrs's outer Newton now finds
    //     a different `λ_2` (371 vs v0.x's 6.49e6) because the matching
    //     formula no longer compensates the unridged-vs-ridged `log|H|`
    //     gap. End-to-end μ-RMSE regresses from 0.0023% → 1.45%.
    //
    // Ergo: rank_adj=0 is the right setting for Tweedie while gamrs's
    // PIRLS factor is unridged (v0.x ridged its score-side factor by
    // `1e-5 · max_diag`). If gamrs ever adopts a score-side ridge, revisit
    // this with the diagnostic harness at
    // `scripts/diagnostics/tweedie_parity_layered.py`.

    /// `log φ` keeps the default ±10. The p axis needs more: near either end
    /// of mgcv's map the score is `∝ e^θ`, so the decrease still on offer
    /// equals the gradient. Stopping at θ = −10 forfeited 0.41 REML units on
    /// Poisson-like data, where p belongs at 1.01; at −30 that is `e^{-20}` of it.
    /// `log|H|` off the Newton weights, as mgcv's extended-family `tw()` does
    /// (log link is not Tweedie's canonical link). Matched-λ, the Fisher
    /// version drifted 3.9e-5 against mgcv's criterion across λ.
    fn use_newton_irls(&self) -> bool {
        true
    }
    fn shape_axis_bounds(&self) -> Vec<(f64, f64)> {
        if self.profile_p {
            vec![(-10.0, 10.0), (-30.0, 30.0)]
        } else {
            vec![(-10.0, 10.0)]
        }
    }
    fn n_shape_params(&self) -> usize {
        // profile-p: [log φ, p_transform]; fixed-p: [log φ] only.
        if self.profile_p {
            2
        } else {
            1
        }
    }
    /// mgcv `build_outer_search_vector`: TweedieLogPhi step cap 1.0,
    /// TweedieP-transform step cap 2.0. Fixed-p drops the p-axis cap.
    fn shape_axis_step_caps(&self) -> Vec<f64> {
        if self.profile_p {
            vec![1.0, 2.0]
        } else {
            vec![1.0]
        }
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(
            params.len(),
            self.n_shape_params(),
            "Tweedie expects {} shape param(s) ({})",
            self.n_shape_params(),
            if self.profile_p {
                "[log φ, p_transform]"
            } else {
                "[log φ] (p fixed)"
            }
        );
        // φ floor — keep > 1e-6 so the series log-W stays well-conditioned.
        self.phi = params[0].exp().max(1e-6);
        // mgcv's `tw()` map keeps p inside [1.01, 1.99], where the Dunn-Smyth
        // series stays finite. It used to be `1 + sigmoid(θ)` clamped to
        // [1.05, 1.95]: past the clamp θ moved and p did not, the score went
        // flat, and the gradient — scaled by dp/dθ at the clamped p — did not.
        // In fixed-p mode there is no p_transform entry.
        if self.profile_p {
            self.p = tweedie_p_from_theta(params[1]);
        }
    }
    fn get_shape_params(&self) -> Vec<f64> {
        if self.profile_p {
            vec![self.phi.ln(), tweedie_theta_from_p(self.p)]
        } else {
            vec![self.phi.ln()]
        }
    }

    /// Analytic Tweedie shape-score gradient — Phase-1 v0.2 port
    /// (2026-05-24). Replaces FD-on-Σ-ls with the v0.x analytical
    /// derivatives of the full Tweedie saturated log-lik
    /// `ls_i = l_base(y; φ, p) - log(y) + log W(y; φ, p)`:
    ///   - `Σ ∂log W/∂ρ` and `Σ ∂log W/∂p` via `tweedie_series`
    ///     (`src/pirls/tweedie.rs`)
    ///   - `Σ ∂l_base/∂φ` and `Σ ∂l_base/∂p` in closed form
    ///     (`src/reml/tweedie_joint.rs::tweedie_dls_dp`)
    ///   - `Σ ∂D/∂p` per obs in closed form (Tweedie deviance derivative
    ///     `src/reml/tweedie_joint.rs::tweedie_dd_level1::dth` with
    ///     `dpth1 = 1`)
    ///
    /// Extended-family REML/LAML score for shape-managed φ (matches
    /// v0.x `src/reml/mod.rs:483` and mgcv `gam.fit5`):
    ///
    /// ```text
    ///   score = Dp/(2φ) - Σ ls + log|H|/2 - log|λS|+/2 - Mp/2·log(2πφ)
    /// ```
    ///
    /// Envelope theorem at converged β̂: `∂β̂/∂shape` contributes 0.
    ///
    /// - log φ: W (PIRLS weights) is φ-free, so `∂log|H|/∂(log φ) = 0`
    ///   exactly. The remaining φ-dependence:
    ///   ```text
    ///     ∂score/∂(log φ) = -Dp/(2φ) - Mp/2 + Σ l_base - Σ dlog_w_drho
    ///   ```
    ///   (since `∂l_base/∂(log φ) = -l_base` — `l_base ∝ 1/φ`, and
    ///   `d(1/φ)/d(log φ) = -1/φ`).
    ///
    /// - p_trans: W depends on p, so `∂log|H|/∂p ≠ 0`. Envelope
    ///   approximation here (the IFT chain via v0.x `tweedie_dd_level1`
    ///   is Phase-2 follow-up). The remaining analytic pieces are:
    ///   ```text
    ///     ∂score/∂(p_trans) ≈ [(1/(2φ))·Σ ∂D/∂p - Σ ∂l_base/∂p
    ///                          - Σ dlog_w_dp] · (p-1)(2-p)
    ///   ```
    fn analytic_shape_score_gradient(
        &self,
        y: ndarray::ArrayView1<f64>,
        mu: ndarray::ArrayView1<f64>,
        dp: f64,
        _n_minus_mp: f64,
        phi_score: f64,
    ) -> Option<ndarray::Array1<f64>> {
        let phi = phi_score.max(1e-12);
        let p = self.p;
        let onep = 1.0 - p;
        let twop = 2.0 - p;
        let profile_p = self.profile_p;
        let inv_onep = 1.0 / onep;
        let inv_twop = 1.0 / twop;
        let inv_onep2 = inv_onep * inv_onep;
        let inv_twop2 = inv_twop * inv_twop;
        let inv_phi = 1.0 / phi;

        // Series derivatives.
        let y_slice: Vec<f64> = y.iter().copied().collect();
        let (_, dlog_w_drho, _, dlog_w_dp) = crate::special::tweedie_series(&y_slice, phi, p);
        let sum_dlog_w_drho: f64 = dlog_w_drho.iter().sum();
        let sum_dlog_w_dp: f64 = dlog_w_dp.iter().sum();

        let n = y.len();
        debug_assert_eq!(mu.len(), n, "y/mu length mismatch in Tweedie analytic grad");

        // Per-obs l_base contributions and its p-derivative
        // (`tweedie_dls_dp::dl_base_dp` from v0.x).
        let mut sum_l_base = 0.0_f64;
        let mut sum_dl_base_dp = 0.0_f64;
        for &yi in y.iter() {
            if yi <= 0.0 {
                continue;
            }
            let log_y = yi.ln();
            let y_2p = yi.powf(twop);
            // l_base = y^(2-p) · (1/(1-p) - 1/(2-p)) · (1/φ)
            //        = y^(2-p) / ((1-p)·(2-p)·φ)
            sum_l_base += y_2p * (inv_onep - inv_twop) * inv_phi;
            // ∂l_base/∂p = y^(2-p) · [−log(y)·(1/(1-p) − 1/(2-p))
            //                         + 1/(1-p)² − 1/(2-p)²] / φ
            sum_dl_base_dp +=
                y_2p * (-log_y * (inv_onep - inv_twop) + inv_onep2 - inv_twop2) * inv_phi;
        }

        // ∂D/∂p per obs (no chain factor — we apply `dp/dp_trans` below).
        // Mirrors v0.x `tweedie_dd_level1` with `dpth1 = 1`.
        let mut sum_dd_dp = 0.0_f64;
        for i in 0..n {
            let yi = y[i];
            let mu_i = mu[i].max(1e-300);
            let log_mu = mu_i.ln();
            let mu1p = mu_i.powf(onep);
            let mu2p = mu_i * mu1p; // μ^(2-p)
            let y_2p = if yi > 0.0 { yi.powf(twop) } else { 0.0 };
            let y_2p_log = if yi > 0.0 { y_2p * yi.ln() } else { 0.0 };
            let y_mu1p = yi * mu1p; // y · μ^(1-p)

            let term_a = (y_2p_log - mu2p * log_mu) * inv_twop;
            let term_b = (y_mu1p * log_mu - y_2p_log) * inv_onep;
            let term_c = -(y_2p - mu2p) * inv_twop2;
            let term_d = (y_2p - y_mu1p) * inv_onep2;
            sum_dd_dp += 2.0 * (term_a + term_b + term_c + term_d);
        }

        // Mp from n_minus_mp = n - Mp.
        let mp = (n as f64) - _n_minus_mp;

        // d/d(log φ) terms:
        //   [Dp/(2φ)]              → -Dp/(2φ)
        //   [-Mp/2·log(2πφ)]       → -Mp/2
        //   [log|H|/2]             → 0  (W is φ-free for Tweedie)
        //   [-Σ ls] where ls = l_base − log y + log W:
        //     [-Σ l_base]/d(log φ) → +Σ l_base   (since l_base ∝ 1/φ)
        //     [-Σ log W]/d(log φ)  → -Σ dlog_w_drho
        let g_log_phi = -dp / (2.0 * phi) - 0.5 * mp + sum_l_base - sum_dlog_w_drho;

        // Fixed-p mode: only the log-φ axis is a shape parameter; p stays
        // constant, so the p-transform gradient component is dropped (the
        // outer Newton never updates p).
        if !profile_p {
            return Some(ndarray::Array1::from_vec(vec![g_log_phi]));
        }

        // d/dp terms (envelope, ignoring ∂log|H|/∂p):
        //   [Dp/(2φ)]    → (1/(2φ)) · Σ ∂D/∂p
        //   [-Mp/2·log(2πφ)] → 0
        //   [-Σ ls]:
        //     [-Σ l_base] → -Σ ∂l_base/∂p
        //     [-Σ log W]  → -Σ dlog_w_dp
        let dp_dpt = tweedie_dp_dtheta(p);
        let g_p_trans = (sum_dd_dp / (2.0 * phi) - sum_dl_base_dp - sum_dlog_w_dp) * dp_dpt;

        Some(ndarray::Array1::from_vec(vec![g_log_phi, g_p_trans]))
    }
}

impl VarianceFn for TweedieVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu.max(1e-300).powf(self.p)
    }
    // Without these the trait's zero defaults dropped V'/V from every
    // observed-curvature weight for Tweedie (newton_score_weights' α).
    fn d_variance(&self, mu: f64) -> f64 {
        self.p * mu.max(1e-300).powf(self.p - 1.0)
    }
    fn d2_variance(&self, mu: f64) -> f64 {
        self.p * (self.p - 1.0) * mu.max(1e-300).powf(self.p - 2.0)
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        if self.profile_p {
            debug_assert_eq!(
                params.len(),
                2,
                "TweedieVariance (profile-p) expects 2 shape params"
            );
            self.p = tweedie_p_from_theta(params[1]);
        } else {
            // Fixed-p: slice is [log φ] only; p stays constant.
            debug_assert_eq!(
                params.len(),
                1,
                "TweedieVariance (fixed-p) expects 1 shape param [log φ]"
            );
        }
    }
}

/// Bounds of the profiled Tweedie power: mgcv `tw(a = 1.01, b = 1.99)`.
pub(crate) const TWEEDIE_P_MIN: f64 = 1.01;
pub(crate) const TWEEDIE_P_MAX: f64 = 1.99;

/// mgcv `tw()`'s map from the unbounded `p_transform` θ to the power:
/// `p = (a + b·e^θ) / (1 + e^θ)`. Smooth all the way to the bounds, so the
/// score flattens there and its derivative goes to zero with it — no clamp, no
/// region where θ moves and p does not.
pub(crate) fn tweedie_p_from_theta(theta: f64) -> f64 {
    let s = 1.0 / (1.0 + (-theta).exp());
    TWEEDIE_P_MIN + (TWEEDIE_P_MAX - TWEEDIE_P_MIN) * s
}

/// Inverse of [`tweedie_p_from_theta`]; `p` outside `(a, b)` maps to the nearest
/// representable end.
pub(crate) fn tweedie_theta_from_p(p: f64) -> f64 {
    let s = ((p - TWEEDIE_P_MIN) / (TWEEDIE_P_MAX - TWEEDIE_P_MIN)).clamp(1e-12, 1.0 - 1e-12);
    (s / (1.0 - s)).ln()
}

/// `dp/dθ` at power `p`: `(p − a)(b − p) / (b − a)`.
pub(crate) fn tweedie_dp_dtheta(p: f64) -> f64 {
    (p - TWEEDIE_P_MIN) * (TWEEDIE_P_MAX - p) / (TWEEDIE_P_MAX - TWEEDIE_P_MIN)
}

/// Phase 9 convenience constructor — **profile-p** Tweedie + log link at
/// given init `(p, φ)` (mgcv `tw()`). `p` is estimated jointly with `φ`
/// and the smoothing params: 2 shape params `[log φ, p_transform]`.
pub fn tweedie_log(p: f64, phi: f64) -> Family<Tweedie, LogLink, TweedieVariance> {
    Family::new(
        Tweedie {
            p,
            phi,
            profile_p: true,
        },
        LogLink,
        TweedieVariance { p, profile_p: true },
    )
}

/// **Fixed-p** Tweedie + log link (mgcv `Tweedie(p=val, link="log")`).
/// `p` is held CONSTANT at the supplied value — only `φ` (and the
/// smoothing params) are estimated. There is a single shape param
/// `[log φ]`; the p-axis is dropped from every shape derivative so the
/// outer Newton never moves `p`.
pub fn tweedie_log_fixed_p(p: f64, phi: f64) -> Family<Tweedie, LogLink, TweedieVariance> {
    Family::new(
        Tweedie {
            p,
            phi,
            profile_p: false,
        },
        LogLink,
        TweedieVariance {
            p,
            profile_p: false,
        },
    )
}
