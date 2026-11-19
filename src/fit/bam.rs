//! mgcv `bam(method = "fREML")` without `discrete`: sweep-and-drop
//! constraints, `initial.sp` on `X'WX`, and `fast.REML.fit`'s Newton over
//! `(ρ, log φ)` from `log φ = log(0.05·var(y))`.
//!
//! Gaussian identity only: `bam` hands every other family to its
//! performance-iteration loop (`bgam.fit`), which this does not reproduce yet.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::design::{Additive, Centring, PreparedDesign};
use crate::error::{GamrsError, Result};
use crate::family::{ConstantVariance, Family, Gaussian, IdentityLink};
use crate::inner::{CholeskySolver, LinearSolver};
use crate::outer::fast_reml::FastRemlNewton;
use crate::score::fast_reml::FastRemlGaussianScore;

use super::driver::{LambdaInit, MgcvInit};
use super::{FittedGam, LinkKind};

/// Fit `design` the way `mgcv::bam(method = "fREML")` does.
pub fn fit_bam(
    _family: Family<Gaussian, IdentityLink, ConstantVariance>,
    design: &Additive,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam> {
    super::check_lengths(x, y, prior_weights)?;
    let prep = design.prepare_with(x, Centring::SweepDrop)?;
    super::fit_identifiable(prep, prior_weights, |prep| {
        fit_gaussian_bam::<CholeskySolver>(prep, x, y, prior_weights)
    })
}

fn fit_gaussian_bam<S: LinearSolver>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam> {
    if prep.constraints.is_some() {
        return Err(GamrsError::InvalidParameter(
            "bam fREML does not support shape-constrained terms".into(),
        ));
    }
    // bam itself switches to gam's REML when there is nothing to smooth.
    if prep.s_list.is_empty() {
        return super::gaussian::fit_gaussian_from_prep::<S>(prep, x, y, weights);
    }
    let n = y.len();
    let m = prep.s_list.len();
    let rho0 = MgcvInit {
        weights: weights.map(|w| w.to_owned()),
    }
    .init(y, &prep.x_design, &prep.s_list);
    let mean = y.sum() / n as f64;
    let var_y = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let mut theta0 = Array1::<f64>::zeros(m + 1);
    theta0.slice_mut(ndarray::s![..m]).assign(&rho0);
    theta0[m] = (0.05 * var_y).ln();

    let score = FastRemlGaussianScore::new(
        &prep.x_design,
        y,
        weights,
        &prep.s_list,
        &prep.rank_s_list,
        prep.mp,
    )?;
    let outer = FastRemlNewton::default().minimize(&score, theta0)?;

    let rho_hat = outer.theta.slice(ndarray::s![..m]).to_owned();
    let fit = score.fit_at(rho_hat.view())?;
    // Sl.postproc scales Vp by rss/(n − edf); the reported scale is exp(log φ̂).
    let vcov = fit.h_inv * (fit.rss / (n as f64 - fit.edf_total));
    Ok(FittedGam {
        beta: fit.beta,
        lambda: rho_hat.mapv(f64::exp),
        rho: rho_hat,
        scale: outer.theta[m].exp(),
        edf_total: fit.edf_total,
        edf_per_term: fit.edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind: LinkKind::Identity,
        shape_params: Array1::zeros(0),
        stats: Default::default(),
    })
}
