//! Gaussian REML with shape-constrained (`sc`) terms — mgcv's `scasm()`.
//!
//! The coefficients must satisfy `A β ≥ 0` (the prepared design's
//! [`ShapeConstraints`]). At fixed λ the penalised least-squares fit under
//! those rows comes from [`pcls`]; the rows binding there hold as
//! equalities.
//!
//! **Which REML.** scasm chooses λ by the REML of the model *reduced* to the
//! binding rows (their directions removed from the coefficients). That
//! criterion jumps wherever a row starts or stops binding, because the two
//! sides count a different number of free coefficients; on a sizeable share
//! of tensor fits its optimum sits on such a seam and it has no stationary
//! point at all. scasm's own iteration then cycles across the seam for good
//! and stops only because its 1% tolerance fires somewhere on the cycle.
//!
//! gamrs instead minimises the REML of the centred (unreduced) model
//! evaluated at the constrained fit — the profiled-scale form of the LAML
//! scasm reports for its fit. It is continuous in ρ with a continuous
//! gradient: the penalised deviance at the constrained minimum has
//! derivative `λ_j βᵀS_jβ` (Danskin's theorem), exactly as unconstrained,
//! and the log-determinants do not depend on β. So the ordinary closed-form
//! score and outer Newton apply as they stand; the only change is an inner
//! solver that returns the constrained coefficients ([`ConstrainedInner`]).
//! Its Hessian uses the unconstrained `dβ/dρ`, which is exact where no row
//! binds and an approximation otherwise; the Newton line search on the exact
//! value and the gradient test keep the search correct either way.
//!
//! Reported quantities: β is the constrained fit; edf comes from the model
//! reduced to its binding rows (those directions are fixed); the scale is
//! `rss / (n − edf)` and the covariance is the centred model's, both as
//! scasm reports them; the REML value is the criterion minimised.

use ndarray::{Array1, Array2, ArrayView1};

use crate::design::{rank_and_log_pseudo_det, PreparedDesign, ShapeConstraints};
use crate::error::{GamrsError, Result};
use crate::family::Gaussian;
use crate::inner::pcls::{null_space, pcls};
use crate::inner::{xtwx_xtwy, DesignMatrix, GaussianClosedFormInner, LinearSolver};
use crate::outer::NewtonWithHalving;
use crate::score::{EnvelopeScore, MgcvTwoSigmaProfile};
use crate::traits::{InnerSolver, OuterSolver, ScoreDerivatives};

use super::gaussian::gaussian_score;
use super::{compute_edf, compute_edf_per_term, compute_vcov, FittedGam, LinkKind};

pub(crate) fn fit_gaussian_constrained<S: LinearSolver>(
    prep: PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam> {
    crate::outer::reject_unsupported_algorithm("gaussian closed-form")?;
    let outer = {
        let score = constrained_score::<S>(&prep, y, weights)?;
        let newton =
            NewtonWithHalving::new(crate::outer::resolve_tuning(&Gaussian).to_newton_opts());
        newton.minimize(&score, Array1::zeros(prep.s_list.len()))?
    };
    let (mut fit, _) = finish::<S>(prep, y, weights, &outer.theta)?;
    fit.n_iters = outer.iterations;
    fit.converged = outer.converged;
    Ok(fit)
}

type ConstrainedScore<'a, S> =
    EnvelopeScore<Gaussian, ConstrainedInner<'a, S>, MgcvTwoSigmaProfile, S>;

/// The Gaussian closed-form REML score of the centred model with the
/// constrained coefficients plugged in.
fn constrained_score<'a, S: LinearSolver>(
    prep: &'a PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<ConstrainedScore<'a, S>> {
    let cons = constraints_of(prep)?;
    let inner = ConstrainedInner::<S> {
        plain: GaussianClosedFormInner::<S>::new(
            prep.x_design.clone(),
            y.to_owned(),
            weights.map(|w| w.to_owned()),
            prep.s_list.clone(),
        ),
        solver: Pcls::new(prep, cons, y, weights),
        y: y.to_owned(),
        weights: weights.map(|w| w.to_owned()),
    };
    Ok(ConstrainedScore::<S>::with_inner(
        inner,
        Gaussian,
        MgcvTwoSigmaProfile,
        y.to_owned(),
        prep.s_list.clone(),
        prep.rank_s_list.clone(),
        prep.mp,
        prep.log_pseudo_det_s_list.clone(),
    ))
}

/// The REML criterion the λ search minimises, at `lambda`.
pub fn reml_at_lambda(
    prep: &PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    lambda: ArrayView1<f64>,
) -> Result<f64> {
    constrained_score::<crate::inner::CholeskySolver>(prep, y, weights)?
        .value(&lambda.mapv(f64::ln))
}

/// The closed-form Gaussian inner solve with the coefficients replaced by
/// the constrained fit; `A = XᵀWX + S_λ` (its factor, log-determinant and
/// inverse) is the unconstrained one.
pub(crate) struct ConstrainedInner<'a, S: LinearSolver> {
    plain: GaussianClosedFormInner<S>,
    solver: Pcls<'a>,
    y: Array1<f64>,
    weights: Option<Array1<f64>>,
}

impl<S: LinearSolver> InnerSolver for ConstrainedInner<'_, S> {
    type Fit = crate::inner::GaussianInnerFit<S>;

    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit> {
        let mut f = self.plain.fit(rho)?;
        let (beta, _) = self.solver.solve(rho)?;
        let mu = self.plain.x_design.dot(&beta);
        let r = &self.y - &mu;
        let rss = match &self.weights {
            Some(w) => r.iter().zip(w).map(|(r, w)| w * r * r).sum(),
            None => r.dot(&r),
        };
        f.beta = beta;
        f.eta = mu.clone();
        f.mu = mu;
        f.rss = rss;
        f.deviance = rss;
        Ok(f)
    }
}

/// The shape-constrained fit at fixed smoothing parameters `lambda` (one
/// per penalty in `prep.s_list`), reported as at the optimum, together
/// with the binding constraint rows (indices into the design's constraint
/// rows, ascending). mgcv's `scasm(..., sp = lambda)`.
pub fn fit_at_lambda(
    prep: PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    lambda: ArrayView1<f64>,
) -> Result<(FittedGam, Vec<usize>)> {
    if lambda.len() != prep.s_list.len() || !lambda.iter().all(|&l| l > 0.0 && l.is_finite()) {
        return Err(GamrsError::InvalidParameter(format!(
            "fit_at_lambda: need {} positive smoothing parameters, got {lambda}",
            prep.s_list.len()
        )));
    }
    finish::<crate::inner::CholeskySolver>(prep, y, weights, &lambda.mapv(f64::ln))
}

/// Everything reported for the fit at `rho` (see the module notes).
fn finish<S: LinearSolver>(
    prep: PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    rho: &Array1<f64>,
) -> Result<(FittedGam, Vec<usize>)> {
    let cons = constraints_of(&prep)?;
    let n = y.len();
    let (beta, active) = Pcls::new(&prep, cons, y, weights).solve(rho)?;
    let reduced = Reduced::new(&prep, cons, &active)?;
    let score = gaussian_score::<S>(
        &reduced.x_design,
        &reduced.s_list,
        &reduced.rank_s_list,
        reduced.mp,
        &reduced.log_pseudo_det_s_list,
        y,
        weights,
    );
    let reduced_fit = score.inner.fit(rho)?;
    let edf = compute_edf(&reduced.x_design, &reduced_fit.a_weights, &reduced_fit);
    let edf_per_term =
        compute_edf_per_term(&reduced.s_list, rho, reduced.x_design.ncols(), &reduced_fit);

    let resid = &y - &prep.x_design.dot(&beta);
    let rss: f64 = match weights {
        Some(w) => resid.iter().zip(w).map(|(r, w)| w * r * r).sum(),
        None => resid.dot(&resid),
    };
    let scale = rss / (n as f64 - edf);

    let full = GaussianClosedFormInner::<S>::new(
        prep.x_design.clone(),
        y.to_owned(),
        weights.map(|w| w.to_owned()),
        prep.s_list.clone(),
    )
    .fit(rho)?;
    let vcov = compute_vcov(&full, scale);

    let reml = constrained_score::<S>(&prep, y, weights)?.value(rho)?;

    let fit = FittedGam {
        beta,
        rho: rho.clone(),
        lambda: rho.mapv(f64::exp),
        scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: 0,
        converged: true,
        reml_value: reml,
        predictor: prep.predictor,
        vcov,
        link_kind: LinkKind::Identity,
        shape_params: Array1::zeros(0),
        stats: score.stats.snapshot(),
    };
    Ok((fit, active))
}

fn constraints_of(prep: &PreparedDesign) -> Result<&ShapeConstraints> {
    prep.constraints
        .as_ref()
        .ok_or_else(|| GamrsError::InvalidParameter("design has no shape constraints".into()))
}

/// The constrained penalised least-squares solve at a given ρ, with
/// `XᵀWX` and `XᵀWy` formed once.
struct Pcls<'a> {
    prep: &'a PreparedDesign,
    cons: &'a ShapeConstraints,
    xtwx: Array2<f64>,
    xtwy: Array1<f64>,
}

impl<'a> Pcls<'a> {
    fn new(
        prep: &'a PreparedDesign,
        cons: &'a ShapeConstraints,
        y: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
    ) -> Self {
        let (xtwx, xtwy) = xtwx_xtwy(&prep.x_design.split(), y, weights);
        Self {
            prep,
            cons,
            xtwx,
            xtwy,
        }
    }

    /// Coefficients and binding rows (ascending) at `rho`.
    fn solve(&self, rho: &Array1<f64>) -> Result<(Array1<f64>, Vec<usize>)> {
        let p = self.prep.x_design.ncols();
        let h = &self.xtwx + &crate::design::combined_s(&self.prep.s_list, rho, p);
        let fit = pcls(&h, &self.xtwy, &self.cons.rows, &self.cons.start)?;
        let mut active = fit.active;
        active.sort_unstable();
        Ok((fit.beta, active))
    }
}

/// The model on `β = Z γ`, `Z` block-diagonal: an orthonormal null-space
/// basis of each constrained term's active rows on that term's
/// coefficients, the identity everywhere else.
struct Reduced {
    x_design: DesignMatrix,
    s_list: Vec<Array2<f64>>,
    rank_s_list: Vec<usize>,
    log_pseudo_det_s_list: Vec<f64>,
    mp: usize,
}

impl Reduced {
    fn new(prep: &PreparedDesign, cons: &ShapeConstraints, active: &[usize]) -> Result<Self> {
        let p = prep.x_design.ncols();
        let z = active_null_space(cons, active, p)?;
        let x_design: DesignMatrix = prep.x_design.dot(&z).into();
        let s_list: Vec<Array2<f64>> = prep.s_list.iter().map(|s| z.t().dot(s).dot(&z)).collect();
        let mut rank_s_list = Vec::with_capacity(s_list.len());
        let mut log_pseudo_det_s_list = Vec::with_capacity(s_list.len());
        for s in &s_list {
            let (r, d) = rank_and_log_pseudo_det(s.view())?;
            rank_s_list.push(r);
            log_pseudo_det_s_list.push(d);
        }
        let mp = crate::design::penalty_null_space_dim(z.ncols(), &s_list, &rank_s_list);
        Ok(Self {
            x_design,
            s_list,
            rank_s_list,
            log_pseudo_det_s_list,
            mp,
        })
    }
}

fn active_null_space(cons: &ShapeConstraints, active: &[usize], p: usize) -> Result<Array2<f64>> {
    // Column ranges kept as identity, and each block's own null space.
    let mut pieces: Vec<((usize, usize), Array2<f64>)> = Vec::new();
    for b in &cons.blocks {
        let rows: Vec<usize> = active
            .iter()
            .copied()
            .filter(|&r| r >= b.rows.0 && r < b.rows.1)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let sub = cons
            .rows
            .select(ndarray::Axis(0), &rows)
            .slice(ndarray::s![.., b.cols.0..b.cols.1])
            .to_owned();
        pieces.push((b.cols, null_space(&sub, b.cols.1 - b.cols.0)?));
    }
    let dropped: usize = active.len();
    let mut z = Array2::<f64>::zeros((p, p - dropped));
    let (mut r, mut c) = (0, 0);
    for ((a, b), zb) in &pieces {
        while r < *a {
            z[[r, c]] = 1.0;
            r += 1;
            c += 1;
        }
        z.slice_mut(ndarray::s![*a..*b, c..c + zb.ncols()])
            .assign(zb);
        r = *b;
        c += zb.ncols();
    }
    while r < p {
        z[[r, c]] = 1.0;
        r += 1;
        c += 1;
    }
    debug_assert_eq!(c, p - dropped);
    Ok(z)
}
