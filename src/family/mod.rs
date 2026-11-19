//! Layer 2 — Loss / Link / VarianceFn families.
//!
//! `deviance_per_obs(y, μ) = (y - μ)²` and `saturated_log_lik(y)` is a
//! constant in y (the σ² in `-½ log(2π σ²)` is profiled out in the REML
//! score for Gaussian, so we drop the constants here — the criterion is
//! invariant to them).
//!
//! `Family<L, K, V>` is a zero-cost aggregator over the three independent
//! traits — `InnerSolver` and `ScoreDerivatives` impls take a `&Family<…>`
//! to plumb all three pieces through one parameter. The aggregator is NOT
//! a trait itself (the plan §3.2 calls it "for InnerSolver convenience,
//! NOT a separate abstraction layer").
//!
//! Split layout: this module is the aggregator surface; each family lives
//! in its own submodule (`gaussian`, `bernoulli`, …, `ocat`, `elf`).
//! Shared `IdentityLink` / `LogLink` / `LogitLink` live in `link`.

use crate::traits::{Link, Loss, VarianceFn};

pub mod bam;
pub mod bernoulli;
pub mod elf;
pub mod gamma;
pub mod gaussian;
pub mod inverse_gaussian;
pub mod link;
pub mod negbin;
pub mod ocat;
pub mod poisson;
pub mod quasi;
pub mod tdist;
pub mod tweedie;

pub use bernoulli::{bernoulli_logit, Bernoulli, BinomialVariance};
pub use elf::{elf_identity, ElfLoss, ElfVariance};
pub use gamma::{gamma_inverse, gamma_log, Gamma, GammaVariance};
pub use gaussian::{gaussian_identity, ConstantVariance, Gaussian};
pub use inverse_gaussian::{inverse_gaussian_log, InverseGaussian, InverseGaussianVariance};
pub use link::{IdentityLink, InverseLink, LogLink, LogitLink};
pub use negbin::{negbin_log, NegBin, NegBinVariance};
pub use ocat::{ocat_identity, ocat_init_theta, OcatLoss, OcatVariance};
pub use poisson::{poisson_log, Poisson, PoissonVariance};
pub use quasi::{quasibinomial_logit, quasipoisson_log, QuasiBinomial, QuasiPoisson};
pub use tdist::{tdist_identity, TDist, TVariance, MIN_DF as TDIST_MIN_DF};
pub use tweedie::{tweedie_log, tweedie_log_fixed_p, Tweedie, TweedieVariance};

// `elf::elf_parts` is crate-private — re-export at the family-module scope
// so internal callers (`inner::ArmijoElfInner`) can use
// `crate::family::elf_parts` exactly as before the split.
pub(crate) use elf::elf_parts;

/// Bundle of Loss + Link + VarianceFn. Phase 0 uses
/// `Family<Gaussian, IdentityLink, ConstantVariance>`; later phases swap
/// `(Bernoulli, LogitLink, BinomialVar)`, `(Tweedie, LogLink, TweedieVar)`, …
///
/// `Clone` is derived so shape-aware scores can rebuild a family per outer
/// probe (architecture-assumptions.md §E2). The Loss/Link/Variance impls
/// themselves must opt into Clone — Phase 0/1 unit structs derive it
/// trivially; TDist's stateful struct uses `#[derive(Clone)]`.
#[derive(Clone)]
pub struct Family<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone> {
    pub loss: L,
    pub link: K,
    pub variance: V,
}

impl<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone> Family<L, K, V> {
    pub fn new(loss: L, link: K, variance: V) -> Self {
        Self {
            loss,
            link,
            variance,
        }
    }

    /// Number of shape parameters owned by the family — delegates to the
    /// loss (single source of truth; variance components mirror the loss's
    /// dimension via the trait's `set_shape_params` contract).
    pub fn n_shape_params(&self) -> usize {
        self.loss.n_shape_params()
    }

    /// Sync shape parameters across Loss and VarianceFn from a single
    /// transformed-θ slice. Call this from the outer Newton after each θ
    /// update and before invoking the inner solver.
    pub fn set_shape_params(&mut self, params: &[f64]) {
        self.loss.set_shape_params(params);
        self.variance.set_shape_params(params);
    }
}
