// Crate-level clippy allowances. The four categories silenced here are
// either (a) too-broad to fix one site at a time without obscuring the
// numerical code, or (b) inherent to the API surface gamrs ships.
//
// - `too_many_arguments` — the inner solvers and shape-aware drivers
//   take 9-16 args because the inner state is genuinely that wide
//   (design + penalties + family + opts + flags). Bundling them into
//   structs hides intent in the score body.
// - `type_complexity` — closure return / trait-object types in
//   src/python.rs (PyO3 dispatch) and `Result<(GaussianInnerFit<S>,
//   Family<L, K, V>)>` are unavoidable in the trait stack.
// - `needless_range_loop` — score/gradient code is uniformly index-
//   based to match the math and stay cheap on per-iter PIRLS hot paths.
// - `doc_overindented_list_items` / `doc_lazy_continuation` — cosmetic
//   rustdoc complaints on existing module docstrings; fixing means
//   tweaking ~30 doc-comment list items with zero behavioural impact.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation
)]

//! gamrs — GAM core, v2 architecture experiment.
//!
//! Phase 0 scope (per `Projects/mgcv_rust/plans/mgcv_rust - v2 Architecture
//! Plan 2026-05-22.md`): Gaussian + identity link + CR spline + sum-to-zero
//! constraint, single smooth, fit end-to-end through the trait stack with
//! minimal parity against mgcv on a few fixtures. No other families, no
//! tensor products, no Python bindings.
//!
//! Layering (top-down):
//!
//! ```text
//! Layer 1   Basis         (basis.rs)      — design matrix + penalties
//! Layer 1.5 BasisTransform (transform.rs) — sum-to-zero, etc.
//! Layer 2   Loss / Link / Variance (family.rs)
//! Layer 3   InnerSolver   (inner.rs)     — β̂(θ) given fixed θ
//! Layer 4   ScoreDerivatives (score.rs)  — REML/LAML criterion + grad
//! Layer 5   OuterSolver   (outer.rs)     — Newton on θ
//! Layer 6   FittedModel / Predictor (fit.rs)
//! ```

pub mod basis;
pub mod design;
pub mod error;
pub mod family;
pub mod fit;
pub mod gamlss;
pub mod inner;
pub mod outer;
pub mod profile;
pub mod score;
pub mod special;
pub mod stats;
pub mod traits;
pub mod transform;

#[cfg(feature = "python")]
pub mod python;

pub use design::{
    combined_s, Additive, AdditivePredictor, Cr, CrPredictor, CrStable, CrStablePredictor,
    DesignStrategy, Predictor, PreparedDesign, Re, RePredictor, TermSpec,
};
pub use error::{GamrsError, Result};
pub use family::{ocat_init_theta, OcatLoss, OcatVariance};
pub use fit::{
    fit, fit_gaulss, fit_shash, fit_with, fit_with_design, fit_with_solver, FamilyFit,
    FamilyFitWithSolver, FitWithProfile, FittedGam, GaulssFit, GaulssOpts, LinkKind, PredictScale,
    ShashGamFit, ShashGamOpts,
};
pub use inner::{CholeskySolver, LinearSolver, LuSolver};
pub use score::{FixedAtOneProfile, MgcvTwoSigmaProfile, OwnedByLossProfile, Profile};
pub use stats::{FitStats, FitStatsSnapshot};
pub use transform::{StableReparam, SumToZero};
