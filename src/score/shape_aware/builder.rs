//! `ShapeInnerBuilder` trait + canonical `PirlsInnerBuilder` /
//! `OcatInnerBuilder` impls. Lets `ShapeAwareEnvelopeScore` pick its
//! inner solver at the type level rather than via per-family score
//! duplication.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::family::{Family, IdentityLink, OcatLoss, OcatVariance};
use crate::inner::{
    CholeskySolver, GaussianInnerFit, LinearSolver, OcatInner, PirlsInner, PirlsOpts,
};
use crate::traits::{InnerSolver, Link, Loss, VarianceFn};

/// Build an `InnerSolver` from a freshly shape-synced family + the score's
/// owned design fields. The score body uses this to rebuild the inner per
/// outer probe (shape params shift every probe; the inner must see the
/// current family).
///
/// Two concrete impls in gamrs:
/// - `PirlsInnerBuilder<S>` (generic over `L, K, V, S`) — drives TDist/scat,
///   Tweedie, NegBin via the standard PIRLS loop.
/// - `OcatInnerBuilder<S>` — drives the ocat extended family via `OcatInner`,
///   constrained to `Family<OcatLoss, IdentityLink, OcatVariance>` (the
///   only valid Loss/Link/Variance triple for ordered categorical).
pub trait ShapeInnerBuilder<
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver = CholeskySolver,
>
{
    type Inner: InnerSolver<Fit = GaussianInnerFit<S>>;

    fn build(
        &self,
        family: Family<L, K, V>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner;

    /// Coefficient of the λ-dependent ridge this inner solver bakes into
    /// the factor it hands the score — i.e. the `c` in
    /// `A = X'WX + Σλ_jS_j + c·max|diag(X'WX + Σλ_jS_j)|·I`.
    ///
    /// It is NOT a solver detail the score can ignore: the score reads
    /// `log|A|` and `tr(A⁻¹S_j)` off that factor, so the analytic
    /// ρ-gradient owes a `∂ridge/∂ρ_j · tr(A⁻¹)/2` term for whatever
    /// ridge is actually in there. Returning the wrong `c` puts a term
    /// growing like λ into a gradient whose true value decays.
    ///
    /// Default `0.0` — `PirlsInner` hands back an **unridged** factor, the
    /// one β̂ was solved with (`linalg.rs::factor_and_solve`). `OcatInner`
    /// overrides.
    fn score_ridge_scale(&self, _n_terms: usize) -> f64 {
        0.0
    }
}

/// Unit-struct builder for `PirlsInner<L, K, V, S>`. The `S` parameter
/// is resolved from the score's `ShapeAwareEnvelopeScore<…, S>` type
/// context — the builder itself stays stateless.
#[derive(Clone, Copy, Default)]
pub struct PirlsInnerBuilder;

impl<L, K, V, S> ShapeInnerBuilder<L, K, V, S> for PirlsInnerBuilder
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver,
{
    type Inner = PirlsInner<L, K, V, S>;

    fn build(
        &self,
        family: Family<L, K, V>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner {
        PirlsInner {
            x_design,
            y,
            prior_weights,
            s_list,
            family,
            opts,
            _solver: PhantomData,
        }
    }
}

/// Unit-struct builder for `OcatInner<S>`. The `S` parameter is resolved
/// from the score's `ShapeAwareEnvelopeScore<…, S>` type context.
#[derive(Clone, Copy, Default)]
pub struct OcatInnerBuilder;

impl<S: LinearSolver> ShapeInnerBuilder<OcatLoss, IdentityLink, OcatVariance, S>
    for OcatInnerBuilder
{
    type Inner = OcatInner<S>;

    fn build(
        &self,
        family: Family<OcatLoss, IdentityLink, OcatVariance>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner {
        OcatInner {
            x_design,
            y,
            prior_weights,
            s_list,
            family,
            opts,
            _solver: PhantomData,
        }
    }

    /// `OcatInner`'s final-pass factor carries the v0.x adaptive ridge
    /// `1e-5·(1 + √n_pen)·max|diag(A_post_pen)|` (`gam_fit5.rs:246-256`),
    /// so the score's ρ-gradient must differentiate it.
    fn score_ridge_scale(&self, n_terms: usize) -> f64 {
        1.0e-5 * (1.0 + (n_terms as f64).sqrt())
    }
}
