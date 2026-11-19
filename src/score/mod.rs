//! Layer 4 — closed-form REML / LAML score, generic over `Loss` family.
//!
//! Single formula handles both:
//!
//! 1. **Profiled σ²** families (Gaussian, Gamma, InverseGaussian) — score
//!    profiles σ̂² out of the marginal likelihood.
//! 2. **Known σ² = 1** families (Bernoulli, Poisson, NegBin) — the
//!    `(n-Mp)/2·log(2πσ²)` term drops, and σ² disappears from the
//!    envelope gradient's `λβ'Sβ/(2σ²)`.
//!
//! Per mgcv `gam.fit3.r:616-617`:
//!
//! ```text
//!   REML(ρ) = Dp/(2σ²) + (n - Mp)/2·log(2πσ²) + log|H|/2
//!             - log|λS|+/2 - Σ ls(y_i)
//! ```
//!
//! where `Dp = D + λβ'Sβ`, `D = Σ w_i · D_i(y_i, μ_i)` is the GLM
//! deviance, `H = X' diag(W_working) X + λS`, and `ls(y_i)` is
//! `Loss::saturated_log_lik`. For known-σ² families the `(n-Mp)/2·log(2πσ²)`
//! term is a constant in λ — we still include it so the reported score is
//! commensurable with mgcv (but its log-gradient is zero so it doesn't
//! affect the optimum).
//!
//! Split layout: this module re-exports the public surface; the actual
//! impls live in:
//! - `profile`      — `Profile` trait + `MgcvTwoSigmaProfile` /
//!                    `FixedAtOneProfile`.
//! - `envelope`     — `EnvelopeScore` (1-D outer Newton, no shape params).
//! - `shape_aware`  — `ShapeAwareEnvelopeScore`, `ShapeInnerBuilder` trait
//!                    + `PirlsInnerBuilder` / `OcatInnerBuilder`.
//!
//! `log|H|` and `tr(H⁻¹·M)` now live as methods on `GaussianInnerFit<S>`
//! — the previous standalone `trace_solve` here and `trace_a_inv_s` in
//! `inner/` (audit finding #4 — two impls of the same op) collapsed into
//! `LinearSolver::trace_a_inv`, dispatched via the `S` backend type.

/// The shape-aware REML score, from its parts. **One definition.**
///
/// ```text
///   REML = Dp/(2φ) − Mp/2·log(2πφ) + log|H|/2 − log|λS|₊/2 − Σ ls(y_i)
/// ```
///
/// This existed as six separate copies — twice in `shape_aware/score.rs`, once
/// in `shape_aware/gradient.rs`, and once in each of the four
/// `evaluate_reml_at_*` Python diagnostics. Six places that have to agree about
/// what the score *is* is the same hazard that cost the 2026-08 scat
/// investigation: the value and the gradient came to disagree about which
/// matrix `log|H|` was taken off, and nothing caught it because each site
/// spelled the formula out for itself. A diagnostic that re-derives the score
/// is not a diagnostic of the score.
pub(crate) fn reml_score_from_parts(
    dp: f64,
    phi: f64,
    mp: usize,
    log_det_h: f64,
    log_det_lambda_s: f64,
    ls_sum: f64,
) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    dp / (2.0 * phi) - 0.5 * (mp as f64) * (two_pi * phi).ln() + 0.5 * log_det_h
        - 0.5 * log_det_lambda_s
        - ls_sum
}

/// `Σ ls(yᵢ; wᵢ)` under prior weights, per [`Loss::weighted_saturated_log_lik`];
/// unweighted, the plain sum.
///
/// [`Loss::weighted_saturated_log_lik`]: crate::traits::Loss::weighted_saturated_log_lik
pub(crate) fn weighted_saturated_log_lik<L: crate::traits::Loss>(
    loss: &L,
    y: &ndarray::Array1<f64>,
    scale: f64,
    prior_w: Option<&ndarray::Array1<f64>>,
) -> f64 {
    match prior_w {
        Some(w) => y
            .iter()
            .zip(w.iter())
            .map(|(&yi, &wi)| loss.weighted_saturated_log_lik(yi, wi, scale))
            .sum(),
        None => y.iter().map(|&yi| loss.saturated_log_lik(yi, scale)).sum(),
    }
}

pub mod envelope;
pub(crate) mod hess_ift;
pub mod penalty_det;
pub mod profile;
pub mod shape_aware;

pub use envelope::{EnvelopeScore, GaussianClosedFormScore};
pub use profile::{FixedAtOneProfile, MgcvTwoSigmaProfile, OwnedByLossProfile, Profile};
pub use shape_aware::{
    OcatInnerBuilder, PirlsInnerBuilder, ShapeAwareEnvelopeScore, ShapeAwareOcatScore,
    ShapeAwarePirlsScore, ShapeAwarePirlsScoreOwnedPhi, ShapeInnerBuilder,
};
