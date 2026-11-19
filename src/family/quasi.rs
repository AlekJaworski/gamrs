//! QuasiPoisson + QuasiBinomial — delegate to Poisson/Bernoulli but profile φ.

use crate::traits::Loss;

use super::bernoulli::{Bernoulli, BinomialVariance};
use super::link::{LogLink, LogitLink};
use super::poisson::{Poisson, PoissonVariance};
use super::Family;

/// QuasiPoisson — identical deviance/variance/link to Poisson but with the
/// dispersion φ **profiled** rather than fixed at 1. Allows for over- or
/// under-dispersion relative to the Poisson model. mgcv's `quasipoisson()`.
///
/// All math delegates to `Poisson`; only `fixed_dispersion()` differs.
#[derive(Clone)]
pub struct QuasiPoisson;

impl Loss for QuasiPoisson {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        Poisson.initial_mu(y)
    }
    fn mgcv_mustart(
        &self,
        y: ndarray::ArrayView1<f64>,
        prior_weights: Option<ndarray::ArrayView1<f64>>,
    ) -> ndarray::Array1<f64> {
        Poisson.mgcv_mustart(y, prior_weights)
    }
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        Poisson.deviance_per_obs(y, mu)
    }
    /// Quasi-likelihood has no true probability density, so the saturated
    /// log-lik is undefined. v0.x and gamrs use mgcv's Gaussian-approximation
    /// `-0.5·log(2πφ)` per obs (sum `-n/2·log(2πφ)`). Phase-2b v0.2 port:
    /// switched FROM the Poisson sat_lik (σ²-independent) TO the Gaussian
    /// approximation so the σ²-profile equation in the new Form B score
    /// `Dp/(2σ²) - Mp/2·log(2πσ²) - Σls` admits the positive root
    /// `σ̂² = Dp/(n-Mp)`. Without this fix the Form B profile would have
    /// NO positive root for σ²-independent quasi sat_lik.
    fn saturated_log_lik(&self, _y: f64, scale: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        -0.5 * (two_pi * scale).ln()
    }
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        Poisson.d_loss_dmu(y, mu)
    }
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        Poisson.d2_loss_dmu(y, mu)
    }
    /// **Profiled** — `None`; `EnvelopeScore` wires `MgcvTwoSigmaProfile`
    /// at construction time (see `fit::fit_quasipoisson_cr`). This is the
    /// ONLY substantive difference from `Poisson`, which wires
    /// `FixedAtOneProfile` (since its `fixed_dispersion() = Some(1.0)`).
    fn fixed_dispersion(&self) -> Option<f64> {
        None
    }
}

/// Phase 4 convenience constructor — QuasiPoisson + log link.
pub fn quasipoisson_log() -> Family<QuasiPoisson, LogLink, PoissonVariance> {
    Family::new(QuasiPoisson, LogLink, PoissonVariance)
}

/// QuasiBinomial — Bernoulli deviance/variance/link but profiled φ.
/// mgcv's `quasibinomial()`.
#[derive(Clone)]
pub struct QuasiBinomial;

impl Loss for QuasiBinomial {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        Bernoulli.initial_mu(y)
    }
    fn mgcv_mustart(
        &self,
        y: ndarray::ArrayView1<f64>,
        prior_weights: Option<ndarray::ArrayView1<f64>>,
    ) -> ndarray::Array1<f64> {
        Bernoulli.mgcv_mustart(y, prior_weights)
    }
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        Bernoulli.deviance_per_obs(y, mu)
    }
    /// Gaussian-approximation sat_lik `-0.5·log(2πφ)` per obs (sum
    /// `-n/2·log(2πφ)`). Phase-2b v0.2 port: switched from Bernoulli (σ²-
    /// independent) to the Gaussian approximation so the new Form B score's
    /// σ²-profile equation has the positive root `σ̂² = Dp/(n-Mp)`. Same
    /// rationale as QuasiPoisson above.
    fn saturated_log_lik(&self, _y: f64, scale: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        -0.5 * (two_pi * scale).ln()
    }
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        Bernoulli.d_loss_dmu(y, mu)
    }
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        Bernoulli.d2_loss_dmu(y, mu)
    }
    fn fixed_dispersion(&self) -> Option<f64> {
        None // PROFILED
    }
}

/// Phase 5 convenience constructor — QuasiBinomial + logit link.
pub fn quasibinomial_logit() -> Family<QuasiBinomial, LogitLink, BinomialVariance> {
    Family::new(QuasiBinomial, LogitLink, BinomialVariance)
}
