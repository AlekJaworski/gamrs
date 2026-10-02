//! Binomial / Bernoulli with logit link.

use crate::traits::{Loss, VarianceFn};

use super::link::LogitLink;
use super::Family;

/// Bernoulli (binary 0/1) likelihood. For binary count data with prior
/// weights `n_i`, the same struct works — `y` is interpreted as the empirical
/// proportion `successes / n_i` and the prior weight is the trial count.
/// Phase 1 only ships Bernoulli (n_i = 1); weighted Binomial rides on the
/// general PIRLS path.
#[derive(Clone)]
pub struct Bernoulli;

#[derive(Clone)]
pub struct BinomialVariance;

impl Loss for Bernoulli {
    /// `μ_i = (y_i + 0.5) / 2`. Keeps μ strictly in (0, 1) even when y is
    /// pure 0/1 — without this, `link(μ) = logit(μ)` would diverge.
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        y.iter().map(|&yi| (yi + 0.5) * 0.5).collect()
    }

    /// `D_i = 2[y log(y/μ) + (1-y) log((1-y)/(1-μ))]` with the 0·log(0) = 0
    /// convention. mgcv's `binomial()$dev.resids`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        // Clamp μ to (ε, 1-ε) to avoid log(0) / divide-by-zero at the
        // PIRLS extremes. ε is mgcv's `binomial()$validmu` threshold.
        let eps = 1e-15;
        let mu = mu.clamp(eps, 1.0 - eps);
        let term_y = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
        let term_1my = if y < 1.0 {
            (1.0 - y) * ((1.0 - y) / (1.0 - mu)).ln()
        } else {
            0.0
        };
        2.0 * (term_y + term_1my)
    }

    /// Saturated log-lik for binary y: `y log y + (1-y) log(1-y)`. For y∈{0,1}
    /// this is identically 0; for fractional y (weighted Binomial) it's
    /// nonzero. The score formula uses `Σ ls(y_i)`.
    fn saturated_log_lik(&self, y: f64, _scale: f64) -> f64 {
        let eps = 1e-15;
        let yc = y.clamp(eps, 1.0 - eps);
        if y > 0.0 && y < 1.0 {
            yc * yc.ln() + (1.0 - yc) * (1.0 - yc).ln()
        } else {
            0.0
        }
    }

    /// `∂D/∂μ = -2(y - μ) / (μ(1-μ))`. Derived from the Bernoulli deviance.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let eps = 1e-15;
        let mu = mu.clamp(eps, 1.0 - eps);
        -2.0 * (y - mu) / (mu * (1.0 - mu))
    }

    /// `∂²D/∂μ² = 2 [μ(1-μ) + (y - μ)(1 - 2μ)] / (μ(1-μ))²`. Used by Newton-
    /// curvature paths; for canonical-link IRLS we only need the Fisher
    /// information `E[∂²D/∂μ²] = 2 / (μ(1-μ))` instead.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let eps = 1e-15;
        let mu = mu.clamp(eps, 1.0 - eps);
        let v = mu * (1.0 - mu);
        2.0 * (v + (y - mu) * (1.0 - 2.0 * mu)) / (v * v)
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0)
    }

    /// Bernoulli + logit link: W = μ(1-μ), stable under small β
    /// perturbations. Eligible for IFT warm-start.
    fn allows_no_refresh(&self) -> bool {
        true
    }
}

impl VarianceFn for BinomialVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu * (1.0 - mu)
    }
    fn d_variance(&self, mu: f64) -> f64 {
        1.0 - 2.0 * mu
    }
    fn d2_variance(&self, _mu: f64) -> f64 {
        -2.0
    }
}

/// Phase 1 convenience constructor — Bernoulli + logit link.
pub fn bernoulli_logit() -> Family<Bernoulli, LogitLink, BinomialVariance> {
    Family::new(Bernoulli, LogitLink, BinomialVariance)
}
