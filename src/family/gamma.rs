//! Gamma with log link (incl. `profile_score_sigma2` Newton).

use crate::traits::{Loss, VarianceFn};

use super::link::{InverseLink, LogLink};
use super::Family;

/// Gamma likelihood for continuous positive responses (waiting times,
/// concentrations, etc.). Canonical link is `1/μ`; mgcv's `Gamma()` and
/// `Gamma(link="log")` are both supported but gamrs ships only the log
/// link for now (most common in practice).
///
/// - Deviance: `D(y, μ) = 2[(y - μ)/μ - log(y/μ)]` for `y > 0`.
/// - Variance: `V(μ) = μ²`.
/// - Dispersion: σ² profiled.
/// - Saturated log-lik: dropped (lgamma/digamma terms in φ are score-relevant
///   for mgcv-exact parity, but the simpler form still gives correct ρ̂; see
///   architecture-assumptions.md §C4-gamma for the gap).
#[derive(Clone)]
pub struct Gamma;

#[derive(Clone)]
pub struct GammaVariance;

impl Loss for Gamma {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        // Gamma μ > 0. Use y itself with a small floor.
        y.iter().map(|&yi| yi.max(1e-3)).collect()
    }

    /// `D(y, μ) = 2[(y - μ)/μ + log(μ/y)]` for `y > 0`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        let y = y.max(1e-300);
        2.0 * ((y - mu) / mu + (mu / y).ln())
    }

    /// Saturated log-lik per obs at μ=y: `-lgamma(1/φ) - log(φ)/φ - 1/φ - log(y)`.
    /// Sum over y gives `n·k(φ) - Σlog(y)` with `k(φ) = -lgamma(1/φ) - log(φ)/φ - 1/φ`.
    ///
    /// Phase-2b v0.2 port (2026-05-24): reinstated the φ-dependent `k(φ)`
    /// piece (previously dropped because the envelope-gradient formula
    /// wasn't σ²-chain aware). Now self-consistent: `profile_score_sigma2`
    /// uses the matching mgcv Newton root, so the score body's σ² satisfies
    /// `∂REML/∂σ²|_{σ²=σ²̂} = 0`, and the envelope gradient at σ²̂ has no
    /// chain-correction term to add. Mirrors v0.x
    /// `src/pirls/mod.rs::Family::Gamma | GammaLog` (lines 513-518).
    fn saturated_log_lik(&self, y: f64, scale: f64) -> f64 {
        let inv_phi = 1.0 / scale;
        let k = -crate::special::log_gamma(inv_phi) - scale.ln() * inv_phi - inv_phi;
        k - y.max(1e-300).ln()
    }

    /// `∂D/∂μ = 2(μ - y) / μ²`.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        2.0 * (mu - y) / (mu * mu)
    }

    /// `∂²D/∂μ² = 2(2y - μ) / μ³`.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        2.0 * (2.0 * y - mu) / (mu * mu * mu)
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        None // PROFILED
    }

    /// Gamma's profile-φ Newton — root-find on the σ²-derivative of the
    /// REML score (Form B / GamFit3):
    ///
    /// ```text
    ///   F(φ) = dp + 2n·[ψ(1/φ) + log φ] + Mp·φ = 0
    ///   F'(φ) = (2n/φ)·[1 - ψ'(1/φ)/φ] + Mp
    /// ```
    ///
    /// Verbatim port from v0.x `src/pirls/mod.rs::Family::estimate_phi_mgcv`
    /// Gamma branch (lines 847-883). Triple guard: per-step relative damping
    /// `max(*0.1).min(*10)`, absolute floor `1e-8`, NaN trap.
    fn profile_score_sigma2(&self, dp: f64, n_obs: usize, n_minus_mp: f64, phi_init: f64) -> f64 {
        let n = n_obs as f64;
        let mp_f = n - n_minus_mp;
        let mut phi = phi_init.max(1e-8);
        let tol_abs = 1e-10 * (dp.abs() + mp_f + 1.0);
        for _ in 0..30 {
            let inv_phi = 1.0 / phi;
            let f = dp + 2.0 * n * (crate::special::digamma(inv_phi) + phi.ln()) + mp_f * phi;
            if f.abs() < tol_abs {
                break;
            }
            let fp = (2.0 * n / phi) * (1.0 - crate::special::trigamma(inv_phi) * inv_phi) + mp_f;
            // Guard against zero / near-zero derivative.
            if fp.abs() < 1e-15 {
                break;
            }
            let delta = -f / fp;
            // Triple guard: relative damping + absolute floor + NaN trap.
            let phi_new = (phi + delta).max(phi * 0.1).min(phi * 10.0).max(1e-8);
            if !phi_new.is_finite() {
                break;
            }
            let converged = (phi_new - phi).abs() < 1e-12 * phi;
            phi = phi_new;
            if converged {
                break;
            }
        }
        phi
    }

    /// Gamma profiles σ̂² via Newton-on-φ (above), NOT the closed-form
    /// `Dp/(n−Mp)`. The analytic outer-Newton Hessian uses
    /// `profile_sigma2_drho_factor` for the IFT chain instead of the
    /// closed-form branch.
    fn profile_sigma2_is_closed_form(&self) -> bool {
        false
    }

    /// IFT chain factor for Gamma's Newton-on-φ profile. Returns
    /// `−1/F'(φ̂)` so the analytic outer-Newton Hessian can form
    /// `∂σ²/∂ρ_i = factor · ∂dp/∂ρ_i = factor · λ_i·β'S_iβ`.
    ///
    /// `F'(φ̂) = (2n/φ̂)[1 - ψ'(1/φ̂)/φ̂] + Mp` — identical to the `fp`
    /// term inside `profile_score_sigma2`'s Newton loop, evaluated at the
    /// converged φ̂. mgcv_rust drops this chain entirely (default-OFF
    /// `MGCV_SIGMA_CHAIN`, `reml_hessian_mgcv_exact_ift` line 2773); gamrs
    /// includes it because the analytic factor is cheap and tightens the
    /// analytic-vs-FD Hessian floor from ~1.7e-2 to ~7.7e-4. Paired with
    /// `skip_w_chain_in_hessian = true` matches mgcv_rust's
    /// `reml_hessian_mgcv_exact_closed_form` shape (no W-chain) but with a
    /// strictly tighter Hessian (the σ² chain is real off-optimum).
    /// Closes the `1 + 2d` FD inner-fits per outer Newton step that the
    /// previous FD-Hessian fallback cost.
    fn profile_sigma2_drho_factor(&self, sigma2: f64, n_obs: usize, mp: usize) -> Option<f64> {
        let phi = sigma2.max(1e-8);
        let inv_phi = 1.0 / phi;
        let n = n_obs as f64;
        let mp_f = mp as f64;
        let fp = (2.0 * n / phi) * (1.0 - crate::special::trigamma(inv_phi) * inv_phi) + mp_f;
        if fp.abs() < 1e-15 {
            return None;
        }
        Some(-1.0 / fp)
    }

    /// Skip the `∂W/∂η` chain term in the analytic Hessian — mgcv_rust's
    /// `reml_hessian_mgcv_exact_closed_form` path (`nn_exploring/src/
    /// smooth.rs:48-53`) treats Gamma as if W were β-independent because
    /// the line-search REML score evaluates on working-response form
    /// (where W IS β-frozen at PIRLS convergence). Without this opt-out,
    /// gamrs's `EnvelopeScore::hess_analytic` would add a `dW/dη` chain
    /// term that doesn't match the score, so the analytic vs FD Hessian
    /// disagree by ~7.7e-4 and Newton loses one iter to step-halving.
    fn skip_w_chain_in_hessian(&self) -> bool {
        true
    }

    fn analytic_hessian_with_newton_chain(&self) -> bool {
        true
    }

    /// Gamma's W = (dμ/dη)²/V = μ²/μ² = 1 (log link) or = 1/μ² (inverse
    /// link) — stable under small β perturbations. Eligible for the
    /// NoRefresh IFT line-search shortcut. Matches mgcv_rust's skip-list
    /// at `gam_optimized.rs:1512-1518` (excludes TDist, Quantile, IG,
    /// Tweedie; Gamma is eligible).
    fn allows_no_refresh(&self) -> bool {
        true
    }

    /// mgcv takes `log|H|` off the Newton (observed) weights whenever the link
    /// is not canonical — `if (family$link==family$canonical) fisher <- TRUE`
    /// (`gam.fit3.r:118`). Gamma's canonical link is the inverse, so with the
    /// log link gamrs's Fisher `log|H|` was a different criterion: matched-λ
    /// against mgcv it drifted ~1e-4 across λ. For the inverse link
    /// α = 1 + (y−μ)(V'/V + g''/g') = 1 exactly, so Newton IS Fisher there.
    fn use_newton_irls(&self) -> bool {
        true
    }
}

impl VarianceFn for GammaVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu * mu
    }
    fn d_variance(&self, mu: f64) -> f64 {
        2.0 * mu
    }
    fn d2_variance(&self, _mu: f64) -> f64 {
        2.0
    }
}

/// Phase 7 convenience constructor — Gamma + log link.
pub fn gamma_log() -> Family<Gamma, LogLink, GammaVariance> {
    Family::new(Gamma, LogLink, GammaVariance)
}

/// Gamma + reciprocal (inverse) link — the canonical link mgcv uses for
/// `family = Gamma()` in R. Reuses the link-free `Gamma` Loss / `μ²`
/// `GammaVariance` and pairs them with the new `InverseLink`. Canonical-
/// pair means Fisher == Newton in PIRLS, so no `use_newton_irls` opt-in
/// is needed.
pub fn gamma_inverse() -> Family<Gamma, InverseLink, GammaVariance> {
    Family::new(Gamma, InverseLink, GammaVariance)
}
