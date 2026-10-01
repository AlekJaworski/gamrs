//! Closed-form Gaussian inner solver: one Cholesky, no IRLS iteration.
//!
//! For Gaussian + identity link, PIRLS degenerates to a single weighted
//! least-squares solve. Working weights are constant (the prior weights,
//! or 1 if uniform), so the normal equations are
//!
//!   `(Xᵀ W X + λ S) β = Xᵀ W y`
//!
//! We factor `A = Xᵀ W X + λ S` once via the type-parameterised
//! `S: LinearSolver` backend (default `CholeskySolver`) and cache the
//! factor for reuse by `ScoreDerivatives`. No iteration is needed —
//! identity link means there's no μ-update loop.
//!
//! `GaussianClosedFormInner<S>` implements `crate::traits::InnerSolver`
//! with `type Fit = GaussianInnerFit<S>` — score impls go through the
//! trait, never the free `gaussian_inner_solve` function (which is kept
//! public only as a building block).

use std::marker::PhantomData;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::Result;
use crate::traits::InnerSolver;

use super::{factor_and_solve, xtwx_xtwy, CholeskySolver, GaussianInnerFit, LinearSolver};

/// Solve `(X'WX + S_total) β = X'Wy` for the combined `S_total = Σ_j
/// λ_j S_j`. `weights` is the prior weight vector; pass `None` for
/// unweighted. Builds `X'WX` / `X'Wy` on-the-fly — the outer Newton hot
/// path uses `GaussianClosedFormInner` instead, which caches both at
/// construction.
///
/// Defaults to the Cholesky backend; for the LU path use
/// `gaussian_inner_solve::<LuSolver>(...)`. Both produce numerically
/// identical β̂ to 1e-13 across the parity battery (the §C4-note Phase-5b
/// hypothesis that the gap was here was empirically invalidated — see
/// `linalg.rs::LuSolver`).
pub fn gaussian_inner_solve<S: LinearSolver>(
    x_design: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    s_total: ArrayView2<f64>,
) -> Result<GaussianInnerFit<S>> {
    let (xtwx, xtwy) = xtwx_xtwy(x_design, y, weights);
    gaussian_inner_solve_cached::<S>(x_design, y, weights, xtwx.view(), xtwy.view(), s_total)
}

/// `crate::traits::InnerSolver` impl for the Gaussian + identity-link path.
/// Caches `X'WX` and `X'Wy` at construction since they don't depend on λ
/// — the outer Newton calls `fit()` ~50-100 times per outer optimisation,
/// and rebuilding them each call was a ~3× perf hit (bench_gaussian
/// confirmed the regression vs v0.x's caching).
///
/// `S: LinearSolver` (default `CholeskySolver`) picks the factorisation
/// backend at the type level.
pub struct GaussianClosedFormInner<S: LinearSolver = CholeskySolver> {
    pub x_design: Array2<f64>,
    pub y: Array1<f64>,
    pub weights: Option<Array1<f64>>,
    /// Per-term penalty blocks. The closed-form solver assembles
    /// `S_total(ρ) = Σ_j exp(ρ_j) · S_j` on each `fit(ρ)` call.
    pub s_list: Vec<Array2<f64>>,
    /// `X' diag(W) X` — cached at construction.
    cached_xtwx: Array2<f64>,
    /// `X' diag(W) y` — cached at construction.
    cached_xtwy: Array1<f64>,
    _solver: PhantomData<S>,
}

impl<S: LinearSolver> GaussianClosedFormInner<S> {
    pub fn new(
        x_design: Array2<f64>,
        y: Array1<f64>,
        weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
    ) -> Self {
        let (cached_xtwx, cached_xtwy) = xtwx_xtwy(
            x_design.view(),
            y.view(),
            weights.as_ref().map(|w| w.view()),
        );
        Self {
            x_design,
            y,
            weights,
            s_list,
            cached_xtwx,
            cached_xtwy,
            _solver: PhantomData,
        }
    }
}

impl<S: LinearSolver> InnerSolver for GaussianClosedFormInner<S> {
    type Fit = GaussianInnerFit<S>;

    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit> {
        debug_assert_eq!(
            rho.len(),
            self.s_list.len(),
            "GaussianClosedFormInner: rho length {} must equal s_list length {}",
            rho.len(),
            self.s_list.len()
        );
        let s_total = crate::design::combined_s(&self.s_list, rho, self.x_design.ncols());
        gaussian_inner_solve_cached::<S>(
            self.x_design.view(),
            self.y.view(),
            self.weights.as_ref().map(|w| w.view()),
            self.cached_xtwx.view(),
            self.cached_xtwy.view(),
            s_total.view(),
        )
    }
}

/// Hot path: solve `(X'WX + λS) β = X'Wy` given pre-cached `X'WX` / `X'Wy`.
/// `lambda` is the ONLY thing that varies between outer probes for the
/// Gaussian path, so factoring this out of `gaussian_inner_solve` lets the
/// outer Newton avoid 200 redundant `X'WX` builds.
fn gaussian_inner_solve_cached<S: LinearSolver>(
    x_design: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    xtwx: ArrayView2<f64>,
    xtwy: ArrayView1<f64>,
    s_total: ArrayView2<f64>,
) -> Result<GaussianInnerFit<S>> {
    let n = x_design.nrows();
    let p = x_design.ncols();

    // A = X'WX + S_total — start from the cached X'WX.
    let mut a = xtwx.to_owned();
    for i in 0..p {
        for j in 0..p {
            a[[i, j]] += s_total[[i, j]];
        }
    }
    let (a_factor, beta) = factor_and_solve::<S>(&a, xtwy)?;

    let mu = x_design.dot(&beta);
    let rss = if let Some(w) = weights {
        let mut s = 0.0;
        for i in 0..n {
            let r = y[i] - mu[i];
            s += w[i] * r * r;
        }
        s
    } else {
        (&y - &mu).iter().map(|&r| r * r).sum::<f64>()
    };

    let eta = mu.clone();
    let working_response = y.to_owned();
    let working_weights = match weights {
        Some(w) => w.to_owned(),
        None => Array1::ones(n),
    };

    Ok(GaussianInnerFit::<S> {
        beta,
        eta,
        // Factor and its weights travel together — see `a_weights`.
        a_weights: working_weights.clone(),
        // No separate expected curvature on this path — edf/vcov keep
        // using `a_factor`, as before. See `GaussianInnerFit::fisher`.
        fisher: None,
        working_weights,
        working_response,
        deviance: rss,
        iterations: 1,
        converged: true,
        mu,
        rss,
        n,
        p,
        a_factor,
        log_det_h_override: None,
        tk_kkt_inputs: None,
        // Gaussian closed-form: working weights are constant in η ⇒ no
        // `∂W/∂η` term in the analytic Hessian.
        dw_deta: None,
        x_design: None,
    })
}
