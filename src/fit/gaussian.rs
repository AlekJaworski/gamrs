//! Gaussian fit helper — closed-form inner + REML outer.
//!
//! `fit_gaussian_from_prep` is the shared Gaussian driver consumed by
//! the `FamilyFitWithSolver` impl for `Gaussian` in `canonical.rs`. Not
//! a public entry point — the canonical surface is `gamrs::fit(...)`.
//!
//! Multi-smooth (94b): the driver consumes `prep.s_list` directly. The
//! outer Newton's θ has length `prep.s_list.len()`; the σ̂² closed form
//! aggregates `Σ_j λ_j β'S_j β` for `Dp = D + Σ_j λ_j β'S_j β`.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use crate::design::PreparedDesign;
use crate::error::Result;
use crate::family::Gaussian;
use crate::inner::{DesignMatrix, GaussianClosedFormInner, LinearSolver};
use crate::outer::NewtonWithHalving;
use crate::score::joint_scale::JointScaleScore;
use crate::score::{EnvelopeScore, MgcvTwoSigmaProfile};
use crate::traits::{InnerSolver, OuterFit, OuterSolver, ScoreDerivatives};

use super::driver::{LambdaInit, MgcvInit};
use super::{compute_edf, compute_edf_per_term, compute_vcov, FittedGam, LinkKind};

/// Gaussian closed-form REML driver. Generic over `S: LinearSolver` so
/// the per-probe factorisation and the emitted `GaussianInnerFit<S>`
/// flow through the caller's choice.
pub(crate) fn fit_gaussian_from_prep<S: LinearSolver>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam> {
    if prep.constraints.is_some() {
        return super::shape_constrained::fit_gaussian_constrained::<S>(prep, y, weights);
    }
    let n = x.nrows();
    let n_terms = prep.s_list.len();

    let rho0 = MgcvInit {
        weights: weights.map(|w| w.to_owned()),
    }
    .init(y, &prep.x_design, &prep.s_list);
    let (score, outer) = gaussian_reml::<S>(
        &prep.x_design,
        &prep.s_list,
        &prep.rank_s_list,
        prep.mp,
        &prep.log_pseudo_det_s_list,
        y,
        weights,
        rho0,
    )?;

    let rho_hat = outer.theta.clone();
    let lambda_hat: Array1<f64> = rho_hat.iter().map(|&r| r.exp()).collect();
    let final_fit = score.inner.fit(&rho_hat)?;
    let edf = compute_edf(&prep.x_design, &final_fit.a_weights, &final_fit);
    let edf_per_term =
        compute_edf_per_term(&prep.s_list, &rho_hat, prep.x_design.ncols(), &final_fit);

    // mgcv Gaussian closed form: σ̂² = Dp/(n - Mp) with
    // Dp = D + Σ_j λ_j β'S_jβ.
    let mut bsb_total = 0.0_f64;
    for j in 0..n_terms {
        let s_beta = prep.s_list[j].dot(&final_fit.beta);
        let bsb_j: f64 = final_fit
            .beta
            .iter()
            .zip(s_beta.iter())
            .map(|(a, b)| a * b)
            .sum();
        bsb_total += lambda_hat[j] * bsb_j;
    }
    let dp = final_fit.rss + bsb_total;
    let n_minus_mp = (n as f64) - (prep.mp as f64);
    let scale = if n_minus_mp > 0.0 {
        dp / n_minus_mp
    } else {
        f64::NAN
    };

    let vcov = compute_vcov(&final_fit, scale);
    Ok(FittedGam {
        beta: final_fit.beta,
        rho: rho_hat,
        lambda: lambda_hat,
        scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind: LinkKind::Identity,
        shape_params: Array1::zeros(0),
        stats: score.stats.snapshot(),
    })
}

/// The Gaussian REML score for one penalised design and its optimum over
/// ρ, searched from `rho0` by the outer Newton. Shared by the plain driver
/// and the shape-constrained one, which runs it on the design reduced to
/// the null space of the binding constraint rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn gaussian_reml<S: LinearSolver>(
    x_design: &DesignMatrix,
    s_list: &[Array2<f64>],
    rank_s_list: &[usize],
    mp: usize,
    log_pseudo_det_s_list: &[f64],
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    rho0: Array1<f64>,
) -> Result<(GaussianScore<S>, OuterFit)> {
    let n_terms = s_list.len();
    let score = gaussian_score::<S>(
        x_design,
        s_list,
        rank_s_list,
        mp,
        log_pseudo_det_s_list,
        y,
        weights,
    );
    // An all-parametric design has no smoothing parameters, so there is nothing for the outer
    // Newton to optimise — its search space is zero-dimensional. Skip it and take the single
    // unpenalised inner fit, which is exactly weighted least squares: with `s_list` empty the
    // penalty sum is zero, so `inner.fit(&[])` already IS the right answer. Reported as converged
    // in 0 iterations because it is closed-form, not a search that happened to stop.
    let outer = if n_terms == 0 {
        OuterFit {
            theta: Array1::zeros(0),
            value: score.value(&Array1::zeros(0))?,
            grad_norm: 0.0, // no parameters, so the gradient is empty and its norm is exactly 0
            iterations: 0,
            converged: true,
        }
    } else {
        crate::outer::reject_unsupported_algorithm("gaussian closed-form")?;
        let tuning = crate::outer::resolve_tuning(&score.loss);
        // Overlapping penalties (tensor margins) keep the default axis-freezing
        // threshold. `design::reparam` cleans their joint null space but not
        // each margin's own, so when one margin heads to λ = ∞ while another
        // stays small, the score past log λ ≈ 20 carries ~1e-5 of rounding
        // noise; following that axis further only descends into the noise
        // (`te_by_covariate_matches_mgcv`). The cure is mgcv's λ-dependent
        // reparameterisation (`gam.reparam`), not yet ported.
        // Search (ρ, log φ) jointly from mgcv's τ₀ = log(null.scale/10),
        // where `null.scale` is the intercept-only model's mean deviance
        // (`estimate.gam`, `get.null.coef`): the same search as mgcv's, so on
        // a multimodal REML it settles in the same minimum. See
        // `score::joint_scale`.
        let n = y.len();
        let null_scale = match weights {
            Some(w) => {
                let mean = y.dot(&w) / w.sum();
                y.iter()
                    .zip(w)
                    .map(|(yi, wi)| wi * (yi - mean).powi(2))
                    .sum::<f64>()
            }
            None => {
                let mean = y.sum() / n as f64;
                y.iter().map(|yi| (yi - mean).powi(2)).sum::<f64>()
            }
        } / n as f64;
        let outer_solver = NewtonWithHalving::new(tuning.to_newton_opts());
        if null_scale > 0.0 && n > mp {
            let joint = JointScaleScore::new(&score, n, mp);
            let joint_solver = NewtonWithHalving::new(crate::outer::NewtonOpts {
                mgcv_indefinite: true,
                ..tuning.to_newton_opts()
            });
            let mut theta0 = Array1::<f64>::zeros(n_terms + 1);
            theta0.slice_mut(s![..n_terms]).assign(&rho0);
            theta0[n_terms] = (null_scale / 10.0).ln();
            let fit = joint_solver.minimize(&joint, theta0)?;
            let rho = fit.theta.slice(s![..n_terms]).to_owned();
            // Report the profiled score at ρ̂ — what mgcv reports, and equal
            // to the joint one once τ has converged to log φ̂.
            OuterFit {
                value: score.value(&rho)?,
                theta: rho,
                ..fit
            }
        } else {
            outer_solver.minimize(&score, rho0)?
        }
    };
    Ok((score, outer))
}

/// The Gaussian closed-form REML score of one penalised design.
pub(crate) fn gaussian_score<S: LinearSolver>(
    x_design: &DesignMatrix,
    s_list: &[Array2<f64>],
    rank_s_list: &[usize],
    mp: usize,
    log_pseudo_det_s_list: &[f64],
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> GaussianScore<S> {
    let inner = GaussianClosedFormInner::<S>::new(
        x_design.clone(),
        y.to_owned(),
        weights.map(|w| w.to_owned()),
        s_list.to_vec(),
    );
    EnvelopeScore::<Gaussian, GaussianClosedFormInner<S>, MgcvTwoSigmaProfile, S>::with_inner(
        inner,
        Gaussian,
        MgcvTwoSigmaProfile,
        y.to_owned(),
        s_list.to_vec(),
        rank_s_list.to_vec(),
        mp,
        log_pseudo_det_s_list.to_vec(),
    )
}

pub(crate) type GaussianScore<S> =
    EnvelopeScore<Gaussian, GaussianClosedFormInner<S>, MgcvTwoSigmaProfile, S>;
