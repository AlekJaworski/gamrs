//! Inverse Gaussian with log link.

use crate::traits::{Loss, VarianceFn};

use super::link::LogLink;
use super::Family;

/// Inverse Gaussian likelihood (a.k.a. Wald) — positive continuous y with
/// variance scaling as `V(μ) = μ³`.
///
/// - Deviance: `D(y, μ) = (y - μ)² / (μ² · y)` for y > 0.
/// - `∂D/∂μ = 2(μ - y) / μ³`.
/// - Variance: `V(μ) = μ³`.
/// - Dispersion: σ² profiled.
#[derive(Clone)]
pub struct InverseGaussian;

#[derive(Clone)]
pub struct InverseGaussianVariance;

impl Loss for InverseGaussian {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        y.iter().map(|&yi| yi.max(1e-3)).collect()
    }

    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        let y = y.max(1e-300);
        let r = y - mu;
        r * r / (mu * mu * y)
    }

    /// Saturated log-lik per obs: `-0.5·log(2π·φ) - 1.5·log y`. Phase-2b
    /// v0.2 port (2026-05-24): reinstated the φ-dependent `-0.5·log(2πφ)`
    /// piece (previously dropped to keep the envelope gradient simple).
    /// Self-consistent now that `EnvelopeScore` uses the σ²_score
    /// throughout (no two-σ² mismatch) and `dls_dsigma2` returns the
    /// matching `-n/(2σ²)` so the profile equation stays correct.
    /// Mirrors v0.x `src/pirls/mod.rs::Family::InverseGaussian` (lines 549-559).
    fn saturated_log_lik(&self, y: f64, scale: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        -0.5 * (two_pi * scale).ln() - 1.5 * y.max(1e-300).ln()
    }

    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        2.0 * (mu - y) / (mu * mu * mu)
    }

    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        2.0 * (3.0 * y - 2.0 * mu) / (mu * mu * mu * mu)
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        None // PROFILED
    }

    /// Inverse Gaussian's canonical link is `1/μ²`; the log link used here
    /// is non-canonical, so mgcv runs full Newton IRLS with per-row Fisher
    /// fallback (`α = 2y/μ - 1 ≤ 0` on ~43% of obs at convergence). Fisher
    /// scoring throughout (the gamrs pre-2026-05-25 default) made `log|H|`
    /// and `tr(H⁻¹S)` in the score body differ from v0.x's Newton-W —
    /// β̂ at the same ρ matches either way (both are stationary points of
    /// the penalised deviance), but the resulting ρ̂ drifted ~1.6e-2 from
    /// mgcv's and propagated to a 3e-4 max-rel-err on μ̂.
    ///
    /// Reference: v0.x `src/pirls/mod.rs::Family::is_canonical_link` marks
    /// `Family::InverseGaussian` non-canonical (lines 464-466), and
    /// `compute_irls_wz` builds the α-corrected (w, z) row pair
    /// accordingly. The 2026-05-25 IG follow-up port enables the same
    /// path in gamrs via this opt-in flag and the new `Link::d2_link_dmu`
    /// + `VarianceFn::d_variance` trait methods.
    fn use_newton_irls(&self) -> bool {
        true
    }
}

impl VarianceFn for InverseGaussianVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu * mu * mu
    }
    fn d_variance(&self, mu: f64) -> f64 {
        // dV/dμ = 3μ²
        3.0 * mu * mu
    }
    fn d2_variance(&self, mu: f64) -> f64 {
        // d²V/dμ² = 6μ
        6.0 * mu
    }
}

/// Phase 8 convenience constructor — InverseGaussian + log link.
pub fn inverse_gaussian_log() -> Family<InverseGaussian, LogLink, InverseGaussianVariance> {
    Family::new(InverseGaussian, LogLink, InverseGaussianVariance)
}
