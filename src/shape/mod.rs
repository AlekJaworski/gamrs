//! Shape-constrained smooths: a port of mgcv's `scasm()` (Gaussian).
//!
//! Smooths use mgcv's `bs` B-spline basis; shape (increasing, decreasing)
//! is a set of linear inequality constraints on the basis coefficients,
//! identifiability a sum-to-zero equality constraint kept in the fit rather
//! than absorbed. Coefficients solve a penalised least squares problem under
//! those constraints (an active-set method, as mgcv's `pcls`), and
//! smoothing parameters come from scasm's extended Fellner-Schall REML loop.

pub mod bspline;
pub mod constraints;
pub mod pcls;
