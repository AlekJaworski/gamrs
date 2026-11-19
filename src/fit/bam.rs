//! mgcv `bam(method = "fREML")` without `discrete`.
//!
//! Every family gets sweep-and-drop constraints and, per working model,
//! `initial.sp` on `X'WX` followed by `fast.REML.fit`'s Newton loop. Gaussian
//! identity is one such fit (`bam.fit`); every other family runs `bgam.fit`'s
//! performance iteration: refit the working model at each IRLS step, halve on
//! the penalised deviance, and re-estimate θ between steps (`estimate.theta`).

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::design::{Additive, Centring, PreparedDesign, TermSpec};
use crate::error::{GamrsError, Result};
use crate::family::bam::BamFamily;
use crate::family::Family;
use crate::inner::{CholeskySolver, DesignMatrix, LinearSolver};
use crate::outer::fast_reml::FastRemlNewton;
use crate::r_rng::RRng;
use crate::score::fast_reml::{r_eigen, FastRemlGaussianScore};
use crate::stats::FitStatsSnapshot;
use crate::traits::{Link, Loss, OuterFit, VarianceFn};

use super::driver::{LambdaInit, MgcvInit};
use super::FittedGam;

const MAX_IRLS: usize = 200;
/// bam's default `chunk.size`: above it, bases are built from a row sample.
const CHUNK_SIZE: usize = 10_000;
const IRLS_EPSILON: f64 = 1e-7;

/// Fit `design` the way `mgcv::bam(method = "fREML")` does.
pub fn fit_bam<L, K, V>(
    family: Family<L, K, V>,
    design: &Additive,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    Family<L, K, V>: BamFamily,
{
    super::check_lengths(x, y, prior_weights)?;
    let w_view = prior_weights.as_ref().map(|w| w.view());
    let prep = match setup_rows(design, x.view(), y.view(), w_view, CHUNK_SIZE) {
        None => design.prepare_with(x, Centring::SweepDrop)?,
        Some(rows) => {
            let mut prep =
                design.prepare_with(x.select(Axis(0), &rows).view(), Centring::SweepDrop)?;
            prep.x_design = DesignMatrix::new(prep.predictor.design(x)?);
            prep
        }
    };
    if prep.constraints.is_some() {
        return Err(GamrsError::InvalidParameter(
            "bam does not support shape-constrained terms".into(),
        ));
    }
    super::fit_identifiable(prep, prior_weights, |prep| {
        if family.is_gaussian_identity() {
            fit_gaussian_bam::<CholeskySolver>(prep, x, y, prior_weights)
        } else {
            fit_pirls_bam(&family, prep, y, prior_weights)
        }
    })
}

fn fit_gaussian_bam<S: LinearSolver>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam> {
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
        link_kind: crate::fit::LinkKind::Identity,
        shape_params: Array1::zeros(0),
        stats: Default::default(),
    })
}

/// One column of bam's model frame, as `mini.mf` sees it.
enum MfColumn<'a> {
    Numeric(ArrayView1<'a, f64>),
    /// Integer-coded factor; levels in increasing code order.
    Factor(ArrayView1<'a, f64>),
}

/// The model frame's columns: response, each variable the terms read (random
/// effects as factors), then the prior weights.
fn model_frame<'a>(
    design: &Additive,
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    w: Option<ArrayView1<'a, f64>>,
) -> Vec<MfColumn<'a>> {
    let mut cols = vec![MfColumn::Numeric(y)];
    let mut seen = Vec::new();
    for t in &design.terms {
        for c in t.cols() {
            if seen.contains(&c) {
                continue;
            }
            seen.push(c);
            let factor = matches!(t, TermSpec::Re { col, .. } if *col == c);
            cols.push(if factor {
                MfColumn::Factor(x.index_axis_move(Axis(1), c))
            } else {
                MfColumn::Numeric(x.index_axis_move(Axis(1), c))
            });
        }
    }
    cols.extend(w.map(MfColumn::Numeric));
    cols
}

/// mgcv's `mini.mf`: the rows bam builds its bases from when there are more
/// than `chunk_size` of them, or `None` to use every row. A sample drawn under
/// `set.seed(66)`, with its first rows overwritten by each numeric column's
/// first minimum and maximum and one row per factor level.
fn setup_rows<'a>(
    design: &Additive,
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    w: Option<ArrayView1<'a, f64>>,
    chunk_size: usize,
) -> Option<Vec<usize>> {
    mini_mf_rows(&model_frame(design, x, y, w), y.len(), chunk_size)
}

fn levels(f: ArrayView1<f64>) -> Vec<f64> {
    let mut l: Vec<f64> = f.to_vec();
    l.sort_by(|a, b| a.total_cmp(b));
    l.dedup();
    l
}

fn mini_mf_rows(cols: &[MfColumn], n: usize, chunk_size: usize) -> Option<Vec<usize>> {
    let min_size: usize = cols
        .iter()
        .map(|c| match c {
            MfColumn::Numeric(_) => 2,
            MfColumn::Factor(f) => levels(*f).len(),
        })
        .sum();
    let chunk_size = chunk_size.max(min_size);
    if n <= chunk_size {
        return None;
    }
    let mut rng = RRng::set_seed(66);
    let mut rows = rng.sample(n, chunk_size);
    let perm = rng.sample(n, n);
    let mut k = 0;
    let first = |v: ArrayView1<f64>, target: f64| v.iter().position(|&a| a == target).unwrap_or(0);
    for c in cols {
        match c {
            MfColumn::Numeric(v) => {
                let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                rows[k] = first(*v, lo);
                rows[k + 1] = first(*v, hi);
                k += 2;
            }
            MfColumn::Factor(f) => {
                for level in levels(*f) {
                    if let Some(&i) = perm.iter().find(|&&i| f[i] == level) {
                        rows[k] = i;
                        k += 1;
                    }
                }
            }
        }
    }
    Some(rows)
}

/// `var(y)` as R computes it (n − 1 denominator).
fn sample_var(y: ArrayView1<f64>) -> f64 {
    let n = y.len() as f64;
    let mean = y.sum() / n;
    y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// One working-model fit: `initial.sp` then `fast.REML.fit`, as `bgam.fit`
/// does at every IRLS step.
struct WorkingFit {
    beta: Array1<f64>,
    rho: Array1<f64>,
    log_phi: f64,
    h_inv: Array2<f64>,
    rss: f64,
    reml_pen: f64,
    outer: OuterFit,
}

fn fit_working(
    prep: &PreparedDesign,
    z: &Array1<f64>,
    w: &Array1<f64>,
    known_scale: Option<f64>,
    log_phi0: f64,
) -> Result<WorkingFit> {
    let m = prep.s_list.len();
    let rho0 = MgcvInit {
        weights: Some(w.clone()),
    }
    .init(z.view(), &prep.x_design, &prep.s_list);
    let score = FastRemlGaussianScore::new(
        &prep.x_design,
        z.view(),
        Some(w.view()),
        &prep.s_list,
        &prep.rank_s_list,
        prep.mp,
    )?
    .with_known_scale(known_scale);
    let theta0 = match known_scale {
        Some(_) => rho0,
        None => {
            let mut t = Array1::<f64>::zeros(m + 1);
            t.slice_mut(s![..m]).assign(&rho0);
            t[m] = log_phi0;
            t
        }
    };
    // With nothing to smooth bam fits each working model by unpenalised least
    // squares (gam's REML with no λ), and its scale optimum rss/(n − p).
    if m == 0 {
        let rho = Array1::zeros(0);
        let fit = score.fit_at(rho.view())?;
        let n = z.len() as f64;
        let log_phi = known_scale.map_or_else(|| (fit.rss / (n - prep.mp as f64)).ln(), f64::ln);
        return Ok(WorkingFit {
            beta: fit.beta,
            reml_pen: score.reml_penalty(rho.view(), log_phi.exp())?,
            log_phi,
            rho,
            h_inv: fit.h_inv,
            rss: fit.rss,
            outer: OuterFit {
                theta: Array1::zeros(0),
                value: f64::NAN,
                grad_norm: 0.0,
                iterations: 0,
                converged: true,
            },
        });
    }
    let outer = FastRemlNewton::default().minimize(&score, theta0)?;
    let rho = outer.theta.slice(s![..m]).to_owned();
    let fit = score.fit_at(rho.view())?;
    let log_phi = known_scale.map_or_else(|| outer.theta[m], f64::ln);
    Ok(WorkingFit {
        beta: fit.beta,
        reml_pen: score.reml_penalty(rho.view(), log_phi.exp())?,
        log_phi,
        rho,
        h_inv: fit.h_inv,
        rss: fit.rss,
        outer,
    })
}

/// `β'S_λβ`.
fn penalty(prep: &PreparedDesign, rho: &Array1<f64>, beta: &Array1<f64>) -> f64 {
    prep.s_list
        .iter()
        .zip(rho.iter())
        .map(|(s_j, r)| r.exp() * beta.dot(&s_j.dot(beta)))
        .sum()
}

fn total_dev<F: BamFamily>(
    family: &F,
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    w: &Array1<f64>,
    theta: &[f64],
) -> f64 {
    (0..y.len())
        .map(|i| family.dev_resid(y[i], mu[i], w[i], theta))
        .sum()
}

/// mgcv's `bgam.fit` for every family but gaussian identity.
fn fit_pirls_bam<F: BamFamily>(
    family: &F,
    prep: PreparedDesign,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
) -> Result<FittedGam> {
    let n = y.len();
    let x: &DesignMatrix = &prep.x_design;
    let pw = prior_weights.map_or_else(|| Array1::ones(n), |w| w.to_owned());
    let known_scale = family.known_scale();
    let mut theta = family.initial_theta(y);
    let extended = !theta.is_empty();

    let mut eta: Array1<f64> = (0..n)
        .map(|i| family.linkfun(family.mustart(y[i], pw[i])))
        .collect();
    let linkinv = |eta: &Array1<f64>| eta.mapv(|e| family.linkinv(e));
    let mut mu = linkinv(&eta);
    let mut dev = 2.0 * total_dev(family, y, &mu, &pw, &theta);

    let mut coef: Option<Array1<f64>> = None;
    let mut last: Option<WorkingFit> = None;
    let (mut coef0, mut eta0, mut pcoef0) = (Array1::zeros(0), Array1::zeros(0), Array1::zeros(0));
    let mut theta0 = theta.clone();
    let mut dev0 = 0.0;
    let mut z = Array1::<f64>::zeros(n);
    let mut wz = Array1::<f64>::zeros(n);
    let mut converged = false;
    let mut iter = 0;
    while iter < MAX_IRLS {
        iter += 1;
        let devold = dev;
        let mut halvings = 0;
        loop {
            if let Some(c) = &coef {
                eta = x.dot(c);
            }
            mu = linkinv(&eta);
            dev = 0.0;
            for i in 0..n {
                let (zi, wi) = family.working(y[i], eta[i], mu[i], pw[i], &theta);
                z[i] = zi;
                wz[i] = wi;
                dev += family.dev_resid(y[i], mu[i], pw[i], &theta);
            }
            if !dev.is_finite() {
                return Err(GamrsError::InvalidParameter(
                    "bam: non-finite deviance".into(),
                ));
            }
            if iter > 2 && (dev - devold).abs() / (0.1 + dev.abs()) < IRLS_EPSILON {
                converged = true;
                break;
            }
            if halvings > 0 {
                break;
            }
            if iter > 2 {
                let fit = last.as_ref().expect("fitted by iteration 2");
                let (dev0_h, mut dev1) = if extended {
                    (
                        total_dev(family, y, &linkinv(&eta0), &pw, &theta0),
                        total_dev(family, y, &mu, &pw, &theta0),
                    )
                } else {
                    (dev0, dev)
                };
                let pen0 = penalty(&prep, &fit.rho, &pcoef0);
                let mut pcoef = fit.beta.clone();
                let mut coef_h = coef.clone().expect("set after the first fit");
                while dev0_h + pen0 < dev1 + penalty(&prep, &fit.rho, &pcoef) && halvings < 6 {
                    coef_h = (&coef0 + &coef_h) / 2.0;
                    pcoef = (&pcoef0 + &pcoef) / 2.0;
                    eta = (&eta0 + &eta) / 2.0;
                    let mu_h = linkinv(&eta);
                    dev = total_dev(family, y, &mu_h, &pw, &theta);
                    dev1 = if extended {
                        total_dev(family, y, &mu_h, &pw, &theta0)
                    } else {
                        dev
                    };
                    halvings += 1;
                }
                coef = Some(coef_h);
            }
            if halvings == 0 {
                break;
            }
        }
        if converged {
            break;
        }
        if iter > 1 {
            coef0 = coef.clone().expect("set after the first fit");
            theta0 = theta.clone();
            pcoef0 = last.as_ref().expect("fitted").beta.clone();
            eta0 = eta.clone();
            dev0 = dev;
        }
        if extended && iter > 1 {
            theta = estimate_theta(family, theta, y, linkinv(&eta).view(), pw.view());
        }
        let log_phi0 = match (&last, coef.is_some()) {
            (Some(f), true) => f.log_phi,
            _ => (0.05 * sample_var(y)).ln(),
        };
        let fit = fit_working(&prep, &z, &wz, known_scale, log_phi0)?;
        coef = Some(fit.beta.clone());
        last = Some(fit);
    }

    let fit = last.expect("at least one working fit");
    // Sl.postproc pairs the last fit's H⁻¹ with the last working weights.
    let xtwx = x.split().xtwx(wz.view());
    let f = fit.h_inv.dot(&xtwx);
    let edf_total = f.diag().sum();
    let edf_per_term = term_edf(&prep, &f);
    let scale = fit.log_phi.exp();
    let vcov_scale = match known_scale {
        Some(phi) => phi,
        None => fit.rss / (n as f64 - edf_total),
    };
    let lambda = fit.rho.mapv(f64::exp);
    // bam reports the family's AIC/2 plus the working fit's penalty terms.
    let mu_hat = linkinv(&x.dot(&fit.beta));
    let reml = family.aic(y, mu_hat.view(), pw.view(), &theta, scale) / 2.0;
    Ok(FittedGam {
        beta: fit.beta,
        lambda,
        rho: fit.rho,
        scale,
        edf_total,
        edf_per_term,
        n,
        n_iters: fit.outer.iterations,
        converged: converged && fit.outer.converged,
        reml_value: reml + fit.reml_pen,
        predictor: prep.predictor,
        vcov: fit.h_inv * vcov_scale,
        link_kind: family.link_kind(),
        shape_params: Array1::from_vec(family.shape_params(&theta)),
        stats: FitStatsSnapshot {
            outer_iterations: fit.outer.iterations,
            inner_pirls_calls: 1,
            inner_pirls_iterations_total: iter,
            ..Default::default()
        },
    })
}

/// Per-term edf: the trace of `H⁻¹X'WX` over each penalised term's columns.
fn term_edf(prep: &PreparedDesign, f: &Array2<f64>) -> Array1<f64> {
    let p = f.nrows();
    prep.s_list
        .iter()
        .map(|s_j| {
            (0..p)
                .filter(|&r| s_j.row(r).iter().any(|&v| v != 0.0))
                .map(|r| f[[r, r]])
                .sum()
        })
        .collect()
}

/// mgcv's `estimate.theta`: Newton on the extended family's negative
/// log-likelihood in θ at fixed μ (scale 1).
fn estimate_theta<F: BamFamily>(
    family: &F,
    mut theta: Vec<f64>,
    y: ArrayView1<f64>,
    mu: ArrayView1<f64>,
    w: ArrayView1<f64>,
) -> Vec<f64> {
    const TOL: f64 = 1e-7;
    let d = theta.len();
    let mut nll = family.theta_nll(y, mu, w, &theta);
    let unconverged = |at: &crate::family::bam::ThetaNll| -> Vec<bool> {
        at.grad
            .iter()
            .map(|g| g.abs() > TOL * (at.value.abs() + 1.0))
            .collect()
    };
    let mut active = unconverged(&nll);
    if !active.iter().any(|&a| a) {
        return theta;
    }
    for _ in 0..100 {
        let idx: Vec<usize> = (0..d).filter(|&i| active[i]).collect();
        let k = idx.len();
        let h = Array2::from_shape_fn((k, k), |(a, b)| nll.hess[[idx[a], idx[b]]]);
        let g = Array1::from_shape_fn(k, |a| nll.grad[idx[a]]);
        let (mut vals, vecs) = match r_eigen(h) {
            Ok(e) => e,
            Err(_) => break,
        };
        if vals.iter().any(|&v| v <= 0.0) {
            vals.mapv_inplace(f64::abs);
            let thresh = vals.iter().fold(0.0_f64, |m, v| m.max(*v)) * 1e-5;
            vals.mapv_inplace(|v| v.max(thresh));
        }
        let mut step0 = -vecs.dot(&(vecs.t().dot(&g) / &vals));
        let ms = step0.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if ms > 4.0 {
            step0.mapv_inplace(|v| v * 4.0 / ms);
        }
        let mut step = vec![0.0; d];
        for (a, &i) in idx.iter().enumerate() {
            step[i] = step0[a];
        }
        let trial =
            |step: &[f64]| -> Vec<f64> { theta.iter().zip(step).map(|(t, s)| t + s).collect() };
        let mut nll1 = family.theta_nll(y, mu, w, &trial(&step));
        let mut halvings = 0;
        let mut failed = false;
        while nll1.value - nll.value > f64::EPSILON.powf(0.75) * nll.value.abs() {
            step.iter_mut().for_each(|s| *s /= 2.0);
            halvings += 1;
            if theta.iter().zip(&step).all(|(t, s)| t + s == *t) || halvings > 25 {
                failed = true;
                break;
            }
            nll1 = family.theta_nll(y, mu, w, &trial(&step));
        }
        if failed {
            break;
        }
        theta = trial(&step);
        nll = nll1;
        active = unconverged(&nll);
        if !active.iter().any(|&a| a) {
            break;
        }
    }
    theta
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::{mini_mf_rows, MfColumn};

    #[derive(serde::Deserialize)]
    struct MiniMf {
        y: Vec<f64>,
        x0: Vec<f64>,
        g: Vec<f64>,
        x1: Vec<f64>,
        w: Vec<f64>,
        chunk_size: usize,
        x0_of_rows: Vec<f64>,
    }

    #[derive(serde::Deserialize)]
    struct Fixture {
        mini_mf: MiniMf,
    }

    #[test]
    fn mini_mf_picks_mgcvs_rows() {
        let fx: Fixture =
            serde_json::from_str(include_str!("../../tests/fixtures/r_rng_and_mini_mf.json"))
                .unwrap();
        let m = fx.mini_mf;
        let (y, x0, g, x1, w) = (
            Array1::from(m.y),
            Array1::from(m.x0),
            Array1::from(m.g),
            Array1::from(m.x1),
            Array1::from(m.w),
        );
        let cols = [
            MfColumn::Numeric(y.view()),
            MfColumn::Numeric(x0.view()),
            MfColumn::Factor(g.view()),
            MfColumn::Numeric(x1.view()),
            MfColumn::Numeric(w.view()),
        ];
        let rows = mini_mf_rows(&cols, y.len(), m.chunk_size).unwrap();
        let got: Vec<f64> = rows.iter().map(|&i| x0[i]).collect();
        assert_eq!(got, m.x0_of_rows);
    }
}
