//! Canonical entry point — `gamrs::fit(family, x, y, weights, k)`.
//!
//! Single typed function dispatches internally to the right driver based
//! on the concrete `Family<L, K, V>` type. Per-family
//! `FamilyFitWithSolver` impls (one per Loss for its (Link, Variance)
//! pair) live in `family_impls.rs`; each impl body is the
//! validate-wire-drive sequence that used to live in a deleted
//! per-family `fit_*_cr_with_solver` wrapper.
//!
//! The `FamilyFit<K, V>` trait wraps `FamilyFitWithSolver<K, V,
//! CholeskySolver>` via a blanket impl, so the no-solver `gamrs::fit(...)`
//! and the solver-parameterised `gamrs::fit_with_solver::<_, _, _,
//! S>(...)` share exactly one dispatch implementation per family (no
//! duplication).
//!
//! Design summary:
//!
//! ```text
//!   pub trait FamilyFitWithSolver<K, V, S>: Loss + Clone + Sized {
//!       fn fit_with_solver_canonical(family, x, y, w, k) -> Result<FittedGam>;
//!   }
//!   pub trait FamilyFit<K, V>: ... { fn fit_canonical(...); }
//!   impl<L, K, V> FamilyFit<K, V> for L where
//!       L: FamilyFitWithSolver<K, V, CholeskySolver> {...}
//! ```
//!
//! Calling `gamrs::fit(gaussian_identity(), x, y, w, 10)` resolves to
//! `<Gaussian as FamilyFitWithSolver<_, _, CholeskySolver>>::
//! fit_with_solver_canonical(...)` at compile time — zero-cost dispatch,
//! no runtime branching, no string keys.
//!
//! `fit_with::<_, _, _, Profile>(...)` lets the 10% of callers who care
//! override the dispersion `Profile` for the Gaussian-family driver.
//! Shape-managed families (Tweedie) and σ²=1 families (Bernoulli,
//! Poisson, NegBin, scat, ocat) don't expose a Profile knob — their
//! convention is baked in by the family.

use ndarray::{ArrayView1, ArrayView2};

use crate::design::{Cr, DesignStrategy, PreparedDesign};
use crate::error::Result;
use crate::family::{ConstantVariance, Family, Gaussian, IdentityLink};
use crate::inner::{CholeskySolver, LinearSolver};
use crate::score::{MgcvTwoSigmaProfile, Profile};
use crate::traits::{Link, Loss, VarianceFn};

use super::FittedGam;

// =============================================================================
// FamilyFitWithSolver — sealed dispatch trait carrying the LinearSolver
// type parameter. Per-family impls live in `family_impls.rs`.
// =============================================================================

/// Sealed dispatch trait — each Loss impls this once for the (Link,
/// Variance) pair it ships with. Drives `gamrs::fit_with_design(...)`,
/// `gamrs::fit(...)` (via the blanket `FamilyFit` impl below) and
/// `gamrs::fit_with_solver::<_, _, _, S>(...)` at the type level — the
/// concrete `Family<L, K, V>` picks which family-specific body to
/// execute without any runtime branching.
///
/// `fit_from_prep_canonical` consumes a `PreparedDesign` produced by
/// any [`DesignStrategy`] — the basis is decoupled from the dispatch
/// trait.
pub trait FamilyFitWithSolver<K, V, S>
where
    Self: Loss + Clone + Sized,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver,
{
    fn fit_from_prep_canonical(
        family: Family<Self, K, V>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam>;
}

// =============================================================================
// FamilyFit — Cholesky-default convenience trait. Blanket-impl over every
// `FamilyFitWithSolver<K, V, CholeskySolver>`; nothing per-family lives
// here.
// =============================================================================

/// Sealed dispatch trait — Cholesky-default flavour of
/// [`FamilyFitWithSolver`]. Used by [`fit`]; identical surface to
/// `FamilyFitWithSolver<K, V, CholeskySolver>`.
pub trait FamilyFit<K, V>
where
    Self: Loss + Clone + Sized,
    K: Link + Clone,
    V: VarianceFn + Clone,
{
    fn fit_from_prep(
        family: Family<Self, K, V>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam>;
}

impl<L, K, V> FamilyFit<K, V> for L
where
    L: FamilyFitWithSolver<K, V, CholeskySolver>,
    K: Link + Clone,
    V: VarianceFn + Clone,
{
    fn fit_from_prep(
        family: Family<Self, K, V>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        super::fit_identifiable(prep, prior_weights, |prep| {
            <L as FamilyFitWithSolver<K, V, CholeskySolver>>::fit_from_prep_canonical(
                family,
                prep,
                x,
                y,
                prior_weights,
            )
        })
    }
}

// =============================================================================
// Public canonical API
// =============================================================================

/// Canonical typed entry point — `gamrs::fit(family, x, y, weights, k)`.
///
/// Dispatches to the appropriate internal driver at compile time based
/// on the concrete `Family<L, K, V>` type:
///
/// - Gaussian / identity → closed-form Gaussian REML.
/// - Bernoulli, Poisson → PIRLS + LAML (σ² ≡ 1).
/// - Quasi{Poisson,Binomial}, Gamma, InverseGaussian → PIRLS + REML
///   with profiled Pearson φ̂ (`MgcvTwoSigmaProfile`).
/// - Tweedie → shape-aware joint Newton on `[ρ, log φ, p_transform]`.
/// - scat (TDist) → shape-aware joint Newton on `[ρ, log σ², log(ν-2)]`.
/// - NegBin → shape-aware joint Newton on `[ρ, log θ]`.
/// - Ocat → shape-aware joint Newton on `[ρ, log-gap thresholds]`.
/// - ELF (quantile) → PIRLS + envelope with σ²=1 and qgam warm-start.
///
/// Default linear backend is `CholeskySolver`; swap to LU with
/// [`fit_with_solver`].
///
/// Errors carry guidance (e.g. `"Gamma requires y > 0; got y=-0.3 at row
/// 42"`); see `GamrsError::InvalidParameter`.
///
/// # Examples
///
/// ```ignore
/// use gamrs::family::{gaussian_identity, tweedie_log};
///
/// // Defaults — automatic dispatch on the family type.
/// let fit = gamrs::fit(gaussian_identity(), x.view(), y.view(), None, 10)?;
///
/// // Shape-managed: initial p, φ live on the family.
/// let fit = gamrs::fit(tweedie_log(1.5, 1.0), x.view(), y.view(), None, 10)?;
/// ```
pub fn fit<L, K, V>(
    family: Family<L, K, V>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    k: usize,
) -> Result<FittedGam>
where
    L: FamilyFit<K, V>,
    K: Link + Clone,
    V: VarianceFn + Clone,
{
    fit_with_design(family, Cr { k }, x, y, prior_weights)
}

/// Canonical typed entry with explicit `DesignStrategy`. Pass `Cr { k }`
/// for the default cubic-regression-spline basis, `Re` for the
/// random-effect (`bs="re"`) basis, or `CrStable { k }` for the
/// stable-reparam variant. Custom strategies are possible — implement
/// the `DesignStrategy` trait — but the produced [`Predictor`] is a
/// closed-set library-controlled enum.
///
/// ```ignore
/// use gamrs::{fit_with_design, design::{Cr, Re}, family::gaussian_identity};
///
/// // CR spline (same as `gamrs::fit(...)`).
/// let fit = fit_with_design(gaussian_identity(), Cr { k: 10 },
///                            x.view(), y.view(), None)?;
///
/// // Random-effect basis (mgcv `bs="re"`).
/// let fit = fit_with_design(gaussian_identity(), Re,
///                            x.view(), y.view(), None)?;
/// ```
pub fn fit_with_design<L, K, V, D>(
    family: Family<L, K, V>,
    design: D,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam>
where
    L: FamilyFit<K, V>,
    K: Link + Clone,
    V: VarianceFn + Clone,
    D: DesignStrategy,
{
    let prep = design.prepare(x)?;
    L::fit_from_prep(family, prep, x, y, prior_weights)
}

/// Canonical entry with explicit `S: LinearSolver` backend. Same family
/// dispatch as [`fit`]; the `S` parameter flows through every inner
/// solver and the emitted `GaussianInnerFit<S>` accessor calls
/// (`log_det_a`, `trace_a_inv`).
///
/// ```ignore
/// use gamrs::{fit_with_solver, family::gaussian_identity, LuSolver};
///
/// // Use LAPACK LU instead of Cholesky for the per-probe factorisation.
/// let fit = fit_with_solver::<_, _, _, LuSolver>(
///     gaussian_identity(), x.view(), y.view(), None, 10
/// )?;
/// ```
///
/// **Empirical finding (2026-05-24):** LU and Cholesky produce β̂
/// identical to 1e-13 on the parity battery — the §C4-note Phase-5b
/// "Cholesky-vs-LU is the gap" hypothesis was invalidated. LU is kept
/// here for forward-compat / v0.x-faithful factor-level parity, not for
/// μ̂ improvement.
pub fn fit_with_solver<L, K, V, S>(
    family: Family<L, K, V>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    k: usize,
) -> Result<FittedGam>
where
    L: FamilyFitWithSolver<K, V, S>,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver,
{
    let prep = Cr { k }.prepare(x)?;
    super::fit_identifiable(prep, prior_weights, |prep| {
        L::fit_from_prep_canonical(family, prep, x, y, prior_weights)
    })
}

// =============================================================================
// fit_with — explicit Profile override (advanced)
// =============================================================================

/// Trait for families that support overriding the dispersion `Profile`
/// at the call site. Implemented only for the 5 profiled-φ Gaussian
/// families. Other families' Profile is baked in (σ² ≡ 1 for Bernoulli /
/// Poisson / NegBin / ocat / ELF / scat; live-loss-owned φ for Tweedie).
///
/// In practice the only meaningful override today is "use a hypothetical
/// `FixedAtOneProfile` on Gaussian" — kept for forward-compat as users
/// experiment with novel Profile impls.
pub trait FitWithProfile<K, V, P>
where
    Self: Loss + Clone + Sized,
    K: Link + Clone,
    V: VarianceFn + Clone,
    P: Profile<Self>,
{
    fn fit_with(
        family: Family<Self, K, V>,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
        k: usize,
    ) -> Result<FittedGam>;
}

// Default Gaussian impl — `MgcvTwoSigmaProfile`. Calling
// `fit_with::<_, _, _, MgcvTwoSigmaProfile>(gaussian(), …)` is therefore
// equivalent to `fit(gaussian(), …)`.
impl FitWithProfile<IdentityLink, ConstantVariance, MgcvTwoSigmaProfile> for Gaussian {
    fn fit_with(
        family: Family<Self, IdentityLink, ConstantVariance>,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
        k: usize,
    ) -> Result<FittedGam> {
        let prep = Cr { k }.prepare(x)?;
        Gaussian::fit_from_prep(family, prep, x, y, prior_weights)
    }
}

/// Canonical entry with explicit `Profile<L>` override. Same as
/// [`fit`] for the families that don't expose Profile knobs (those
/// only impl `FamilyFit`, not `FitWithProfile`).
///
/// Today this only surfaces a knob for Gaussian; the rest is forward-
/// compat for users who want to experiment with custom Profile impls.
///
/// ```ignore
/// use gamrs::{fit_with, family::gaussian_identity, MgcvTwoSigmaProfile};
///
/// let fit = fit_with::<_, _, _, MgcvTwoSigmaProfile>(
///     gaussian_identity(), x.view(), y.view(), None, 10
/// )?;
/// ```
pub fn fit_with<L, K, V, P>(
    family: Family<L, K, V>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    k: usize,
) -> Result<FittedGam>
where
    L: FitWithProfile<K, V, P>,
    K: Link + Clone,
    V: VarianceFn + Clone,
    P: Profile<L>,
{
    L::fit_with(family, x, y, prior_weights, k)
}
