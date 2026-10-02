//! Poisson with log link.
//!
//! - Deviance: `D(y, μ) = 2[y log(y/μ) - (y - μ)]` for `y > 0`, `2μ` for `y = 0`.
//! - Variance: `V(μ) = μ`.
//! - Dispersion: σ² = 1 (fixed).
//! - Saturated log-likelihood: we omit the `lgamma(y + 1)` term (constant in
//!   θ, doesn't affect the λ-optimum or predictions). The reported
//!   `reml_value` will therefore differ from mgcv's by a y-only constant,
//!   but fitted β / λ̂ / predictions match.

use crate::traits::{Loss, VarianceFn};

use super::link::LogLink;
use super::Family;

/// Poisson likelihood for count data. Canonical link is log.
#[derive(Clone)]
pub struct Poisson;

#[derive(Clone)]
pub struct PoissonVariance;

impl Loss for Poisson {
    /// `μ_i = max(y_i, 0.1)` — mgcv's `poisson()` default. Floor at 0.1
    /// keeps `link(μ) = log μ` from diverging when `y = 0`.
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        y.iter().map(|&yi| yi.max(0.1)).collect()
    }

    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        // Clamp μ away from 0 — at μ=0 the log diverges. ε is mgcv's
        // `poisson()$validmu` floor.
        let eps = 1e-300;
        let mu = mu.max(eps);
        if y > 0.0 {
            2.0 * (y * (y / mu).ln() - (y - mu))
        } else {
            2.0 * mu
        }
    }

    /// `ls(y) = y log(y) - y` (omitting `lgamma(y+1)` — constant in θ).
    /// For y=0 the limit is 0. φ is fixed at 1 so the `_scale` arg is moot.
    fn saturated_log_lik(&self, y: f64, _scale: f64) -> f64 {
        if y > 0.0 {
            y * y.ln() - y
        } else {
            0.0
        }
    }

    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        // Derivative of 2[y log(y/μ) - (y - μ)] wrt μ:
        //   ∂/∂μ [-2y log μ + 2μ] = -2y/μ + 2 = 2(μ - y)/μ
        // At y=0 this gives 2 (consistent with the y=0 branch's `2μ` → `∂=2`).
        let eps = 1e-300;
        let mu = mu.max(eps);
        2.0 * (mu - y) / mu
    }

    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        // ∂/∂μ [2(μ - y)/μ] = 2y/μ²
        let eps = 1e-300;
        let mu = mu.max(eps);
        2.0 * y / (mu * mu)
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0)
    }

    /// Poisson + log link: W = μ, stable under small β perturbations.
    /// Eligible for IFT warm-start in line-search trials (matches
    /// mgcv_rust's skip-list at `gam_optimized.rs:1512-1518`).
    fn allows_no_refresh(&self) -> bool {
        true
    }
}

impl VarianceFn for PoissonVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu
    }
    fn d_variance(&self, _mu: f64) -> f64 {
        1.0
    }
}

/// Phase 3 convenience constructor — Poisson + log link.
pub fn poisson_log() -> Family<Poisson, LogLink, PoissonVariance> {
    Family::new(Poisson, LogLink, PoissonVariance)
}
