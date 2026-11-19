//! `Profile` trait + dispersion impls (MgcvTwoSigma, FixedAtOne).
//!
//! Closes architecture-assumptions.md §D3 properly: the choice of "which
//! σ² is THE σ² for the score, and which for the gradient" is a *type*
//! parameter on `EnvelopeScore`, not a runtime match in the score body.
//!
//! The two convention names map to the two profile impls:
//!
//!   MgcvTwoSigmaProfile — Dp/(n-Mp) for score, D/(n - tr(H⁻¹X'WX)) for
//!                         gradient. Mirrors mgcv exactly for the 5
//!                         profiled GLM families (Gaussian, QuasiPoisson,
//!                         QuasiBinomial, Gamma, InverseGaussian).
//!
//!   FixedAtOneProfile   — σ² ≡ 1 for both. Used by Bernoulli, Poisson,
//!                         NegBin — anything whose family deviance is
//!                         already in "natural units" with σ² = 1.
//!
//! The previous runtime `match loss.fixed_dispersion() { Some(phi) => …,
//! None => … }` dispatch is gone; the choice is made at score
//! construction time and the compiler verifies it.

use crate::inner::{GaussianInnerFit, LinearSolver};
use crate::traits::Loss;

/// Layer-4 sub-trait — names *which* σ² convention the score body uses.
///
/// Two-σ² is mgcv's convention (deliberate for parity, see §D3) — score
/// formula and gradient formula sit on different σ̂² estimates. This trait
/// lifts that choice from a buried `match` into a type parameter, so a
/// `EnvelopeScore<L, I, MgcvTwoSigmaProfile>` cannot drift into the wrong
/// convention at runtime.
///
/// `L: Loss` is a type parameter because the profile σ̂² depends on the
/// family's sat_lik formula (Gaussian: closed form; Gamma: Newton root).
/// `Loss::profile_score_sigma2` returns the right answer for each family;
/// the Profile impl just plumbs it through.
///
/// The `dispersion` method is generic over `S: LinearSolver` so the
/// inner-fit handle can come from any backend (the dispersion calc only
/// reads `inner.n` / `inner.deviance` — backend-agnostic).
pub trait Profile<L: Loss> {
    /// Compute the score-side σ̂² from the converged inner fit. Returns
    /// `None` to signal an invalid probe (negative dispersion,
    /// `n - Mp ≤ 0`, etc.) — caller bails to a sentinel score value.
    ///
    /// Phase-2b dropped the mgcv two-σ² convention (score and gradient
    /// now use the SAME σ²); the trait surface returns a single value.
    fn dispersion<S: LinearSolver>(
        &self,
        loss: &L,
        inner: &GaussianInnerFit<S>,
        lambda: f64,
        bsb: f64,
        tr_hinv_xtwx: f64,
        mp: usize,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<f64>;

    /// Analytic `∂σ²_score/∂ρ_i` for the outer-Newton Hessian.
    ///
    /// The envelope gradient is `g[j] = λ_j·bSb_j/(2σ²) + …`, so the
    /// analytic Hessian needs the chain term `−(λ_j·bSb_j/(2σ⁴))·∂σ²/∂ρ_i`.
    /// For the closed-form `σ² = Dp/(n−Mp)` profiles the cancellation
    /// `∂Dp/∂ρ_i = λ_i·bSb_i` (penalised-deviance minimum / normal-equation
    /// envelope identity) gives `∂σ²/∂ρ_i = λ_i·bSb_i/(n−Mp)`. For σ²≡1
    /// families it is identically zero.
    ///
    /// Returns `None` when no closed form is available (e.g. Gamma's
    /// Newton-on-φ profile) — the caller then keeps the finite-difference
    /// Hessian for that family, never silently shipping a wrong analytic one.
    ///
    /// `bsb_per_term[i] = β'S_iβ`, `lambda_j[i] = exp(ρ_i)`. The default
    /// returns `None` (conservative: opt in per profile).
    fn dispersion_drho<S: LinearSolver>(
        &self,
        _loss: &L,
        _inner: &GaussianInnerFit<S>,
        _sigma2: f64,
        _bsb_per_term: &[f64],
        _lambda_j: &[f64],
        _mp: usize,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<Vec<f64>> {
        None
    }
}

/// mgcv's two-σ² REML convention. Used by all profiled-φ GLM families.
///
/// - `score_sigma2 = Loss::profile_score_sigma2(dp, n, n-Mp, …)`
///   — Gaussian closed form `Dp/(n-Mp)` for most families, Newton on
///   `F(φ) = dp + 2n[ψ(1/φ) + log φ] + Mp·φ` for Gamma.
/// - `grad_sigma2 = D / (n - tr(H⁻¹X'WX))` — mgcv `RemlScoreParts::gaussian_only`.
///
/// `D` here is the GLM family deviance from PIRLS convergence (NOT the
/// working-RSS, which vanishes for non-Gaussian families). Returns `None`
/// if the resulting σ̂² is non-positive — the score body uses that as a
/// "this probe is unphysical, bail" signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MgcvTwoSigmaProfile;

impl<L: Loss> Profile<L> for MgcvTwoSigmaProfile {
    fn dispersion<S: LinearSolver>(
        &self,
        loss: &L,
        inner: &GaussianInnerFit<S>,
        lambda: f64,
        bsb: f64,
        tr_hinv_xtwx: f64,
        mp: usize,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<f64> {
        let n_minus_mp = (inner.n as f64) - (mp as f64);
        let n_eff = (inner.n as f64) - tr_hinv_xtwx;
        let dp = inner.deviance + lambda * bsb;
        if n_minus_mp <= 0.0 || n_eff <= 0.0 || dp <= 0.0 || inner.deviance <= 0.0 {
            return None;
        }
        // Family-specific profile σ̂² (Gamma → Newton-on-φ, others → closed
        // form). Warm-start with the simple closed-form value.
        let phi_init = dp / n_minus_mp;
        let score_sigma2 = loss.profile_score_sigma2(dp, inner.n, n_minus_mp, phi_init, prior_w);
        Some(score_sigma2.max(1e-8))
    }

    fn dispersion_drho<S: LinearSolver>(
        &self,
        loss: &L,
        inner: &GaussianInnerFit<S>,
        _sigma2: f64,
        bsb_per_term: &[f64],
        lambda_j: &[f64],
        mp: usize,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<Vec<f64>> {
        let n_minus_mp = (inner.n as f64) - (mp as f64);
        if n_minus_mp <= 0.0 {
            return None;
        }
        // Closed-form profiles (Gaussian / Quasi*): σ² = Dp/(n−Mp), and the
        // envelope identity ∂Dp/∂ρ_i = λ_i·β'S_iβ gives
        // ∂σ²/∂ρ_i = λ_i·bSb_i/(n−Mp).
        if loss.profile_sigma2_is_closed_form() {
            return Some(
                lambda_j
                    .iter()
                    .zip(bsb_per_term.iter())
                    .map(|(&lam, &bsb)| lam * bsb / n_minus_mp)
                    .collect(),
            );
        }
        // Newton-on-φ profiles (Gamma): use the family's analytic IFT
        // factor `−1/F'(φ̂)` so the chain `∂σ²/∂ρ_i = factor · λ_i·bSb_i`
        // is exact, not FD. mgcv_rust treats σ̂² as a constant in the
        // analytic Hessian formula (`reml_hessian_mgcv_exact_ift` at
        // `nn_exploring/src/reml/mod.rs:2511`, line 2773); we improve on
        // that with the cheap analytic factor here, avoiding the `1 + 2d`
        // FD inner-fits per outer Newton step that the previous FD-Hessian
        // fallback cost. Falls back to "treat as constant" (zeros) if the
        // family declines to provide a factor (e.g. degenerate `F'(φ̂)≈0`).
        let factor = loss.profile_sigma2_drho_factor(_sigma2, inner.n, mp, prior_w);
        let factor = match factor {
            Some(f) => f,
            None => return Some(vec![0.0; bsb_per_term.len()]),
        };
        Some(
            lambda_j
                .iter()
                .zip(bsb_per_term.iter())
                .map(|(&lam, &bsb)| factor * lam * bsb)
                .collect(),
        )
    }
}

/// σ² fixed at 1.0 for both score and gradient. Used by canonical GLM
/// families with known unit dispersion: Bernoulli, Poisson, NegBin.
///
/// Score formula collapses to LAML: `D/2 + log|H|/2 - log|λS|+/2 - Σ ls`.
/// Gradient collapses to `λβ'Sβ/2 + λ·tr(H⁻¹S)/2 - rank/2`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedAtOneProfile;

impl<L: Loss> Profile<L> for FixedAtOneProfile {
    fn dispersion<S: LinearSolver>(
        &self,
        _loss: &L,
        _inner: &GaussianInnerFit<S>,
        _lambda: f64,
        _bsb: f64,
        _tr_hinv_xtwx: f64,
        _mp: usize,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<f64> {
        Some(1.0)
    }

    fn dispersion_drho<S: LinearSolver>(
        &self,
        _loss: &L,
        _inner: &GaussianInnerFit<S>,
        _sigma2: f64,
        bsb_per_term: &[f64],
        _lambda_j: &[f64],
        _mp: usize,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<Vec<f64>> {
        // σ² ≡ 1 is constant in ρ — no chain term.
        Some(vec![0.0; bsb_per_term.len()])
    }
}

/// σ² read directly off the family's live state via
/// `Loss::fixed_dispersion().unwrap_or(1.0)`. Used by shape-managed
/// dispersion families (Tweedie) where the dispersion is itself one of
/// the outer-Newton shape parameters — the outer probes set `loss.phi`
/// per probe and the score body needs to read whatever the family
/// currently holds.
///
/// Score and gradient σ² are identical here: the dispersion has already
/// been updated *outside* the score body, so there's no two-σ²
/// distinction to make. Closes audit §80 — replaces the prior runtime
/// `family.loss.fixed_dispersion().unwrap_or(1.0)` branch in
/// `ShapeAwareEnvelopeScore`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OwnedByLossProfile;

impl<L: Loss> Profile<L> for OwnedByLossProfile {
    fn dispersion<S: LinearSolver>(
        &self,
        loss: &L,
        _inner: &GaussianInnerFit<S>,
        _lambda: f64,
        _bsb: f64,
        _tr_hinv_xtwx: f64,
        _mp: usize,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<f64> {
        Some(loss.fixed_dispersion().unwrap_or(1.0))
    }
}
