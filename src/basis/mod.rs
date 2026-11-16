//! Basis functions (Layer 1 of the v2 architecture, plan §3).
//!
//! Each submodule provides one concrete `Basis` impl. Public re-exports keep
//! the old `crate::basis::CrSpline` path working for callers that pre-date
//! the split.

pub mod bspline;
pub mod cr;
pub mod re;
pub mod tensor;

pub use bspline::BsBasis;
pub use cr::CrSpline;
pub use re::RandomEffectsBasis;
pub use tensor::TensorProductBasis;
