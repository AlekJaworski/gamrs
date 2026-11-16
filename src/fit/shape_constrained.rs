//! Gaussian REML with shape-constrained (`sc`) terms — mgcv's `scasm()`.
//!
//! The coefficients must satisfy `A β ≥ 0` (the prepared design's
//! [`ShapeConstraints`]). At fixed λ the penalised least-squares problem is
//! solved under those rows by [`pcls`]; its binding rows (the active set)
//! hold as equalities there, so near that λ the fit is the ordinary
//! penalised fit on the coefficients `β = Z γ` with `Z` an orthonormal basis
//! of the active rows' null space. That reduced model is an ordinary
//! Gaussian GAM (design `X Z`, penalties `Zᵀ S_j Z`), so its REML optimum
//! over ρ is found by the ordinary closed-form score and outer Newton. This
//! is the projection scasm applies (`getZ`, per smooth), with gamrs's
//! Newton in place of scasm's Fellner–Schall updates.
//!
//! The active set can change as λ moves, so the search alternates: fit the
//! constraints at the current ρ, optimise ρ on the model reduced to that
//! fit's active set, refit, and stop once the active set at the new optimum
//! is the one it was optimised for. A bounded number of rounds; a set that
//! never settles is an error, not a silent answer.
//!
//! Reported quantities follow scasm: β is the constrained fit; edf comes
//! from the reduced model (the active rows fix those directions); the scale
//! is `rss / (n − edf)`, which equals the REML profile scale at the optimum;
//! the covariance and the REML value are projected on the centring only
//! (here: the absorbed basis), not on the active rows.

use ndarray::{Array1, Array2, ArrayView1};

use crate::design::{rank_and_log_pseudo_det, PreparedDesign, ShapeConstraints};
use crate::error::{GamrsError, Result};
use crate::inner::pcls::{null_space, pcls};
use crate::inner::{xtwx_xtwy, DesignMatrix, GaussianClosedFormInner, LinearSolver};
use crate::traits::InnerSolver;

use super::gaussian::{gaussian_reml, gaussian_score};
use super::{compute_edf, compute_edf_per_term, compute_vcov, FittedGam, LinkKind};

/// Rounds of "fit the constraints, optimise ρ on the active set" before
/// giving up. Each round after the first starts from the previous optimum,
/// so a settled set is usually confirmed in two or three.
const MAX_ROUNDS: usize = 20;

pub(crate) fn fit_gaussian_constrained<S: LinearSolver>(
    prep: PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam> {
    let cons = constraints_of(&prep)?;
    let solver = Pcls::new(&prep, cons, y, weights);
    let mut rho = Array1::<f64>::zeros(prep.s_list.len());
    let mut active = solver.solve(&rho)?.1;
    let mut iterations = 0;
    for _ in 0..MAX_ROUNDS {
        let reduced = Reduced::new(&prep, cons, &active)?;
        let (_, outer) = gaussian_reml::<S>(
            &reduced.x_design,
            &reduced.s_list,
            &reduced.rank_s_list,
            reduced.mp,
            &reduced.log_pseudo_det_s_list,
            y,
            weights,
            rho.clone(),
        )?;
        rho = outer.theta;
        iterations += outer.iterations;
        let next = solver.solve(&rho)?.1;
        if next == active {
            let (mut fit, _) = finish::<S>(prep, y, weights, &rho)?;
            fit.n_iters = iterations;
            fit.converged = outer.converged;
            return Ok(fit);
        }
        active = next;
    }
    Err(GamrsError::InvalidParameter(format!(
        "shape-constrained fit: the set of binding constraints did not settle in \
         {MAX_ROUNDS} rounds"
    )))
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

/// Everything reported for the fit at `rho`: β from the constrained solve,
/// edf from the model reduced to its binding rows, `scale = rss/(n − edf)`,
/// and the covariance and REML value of the centred (unreduced) model.
fn finish<S: LinearSolver>(
    prep: PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    rho: &Array1<f64>,
) -> Result<(FittedGam, Vec<usize>)> {
    let cons = constraints_of(&prep)?;
    let n = y.len();
    let p = prep.x_design.ncols();
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

    // scasm's REML (LAML) at the fit, on the centred basis:
    // (−Dp/φ + log|S_λ|₊ − log|XᵀWX + S_λ|)/2 + M log(2π)/2 − n log(2πφ)/2.
    let s_lambda = crate::design::combined_s(&prep.s_list, rho, p);
    let (rank_s, log_det_s) = rank_and_log_pseudo_det(s_lambda.view())?;
    let dp = rss + beta.dot(&s_lambda.dot(&beta));
    let two_pi = 2.0 * std::f64::consts::PI;
    let laml = (-dp / scale + log_det_s - full.log_det_a()) / 2.0
        + (p - rank_s) as f64 * two_pi.ln() / 2.0
        - n as f64 * (two_pi * scale).ln() / 2.0;

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
        reml_value: -laml,
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

/// The REML criterion the λ search minimises, at `lambda`: the ordinary
/// Gaussian REML score of the model reduced to the binding rows of the
/// constrained fit there, and those rows. Comparable across λ only where
/// the binding rows agree.
pub fn search_reml_at_lambda(
    prep: &PreparedDesign,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    lambda: ArrayView1<f64>,
) -> Result<(f64, Vec<usize>)> {
    use crate::traits::ScoreDerivatives;
    let cons = constraints_of(prep)?;
    let rho = lambda.mapv(f64::ln);
    let (_, active) = Pcls::new(prep, cons, y, weights).solve(&rho)?;
    let r = Reduced::new(prep, cons, &active)?;
    let score = gaussian_score::<crate::inner::CholeskySolver>(
        &r.x_design,
        &r.s_list,
        &r.rank_s_list,
        r.mp,
        &r.log_pseudo_det_s_list,
        y,
        weights,
    );
    Ok((score.value(&rho)?, active))
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
