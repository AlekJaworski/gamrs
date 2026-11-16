//! Gaussian fit of a shape-constrained additive model, as mgcv's `scasm()`
//! does it: coefficients by [`pcls`](super::pcls::pcls) at given smoothing
//! parameters, smoothing parameters by scasm's extended Fellner-Schall
//! (EFS) loop on the REML fixed point, with the Hessian and penalties
//! projected onto the null space of the sum-to-zero rows and of the
//! inequality constraints active at the current fit (`scasm.fit`,
//! `getVb`, `trSiSZ`, `getZ`).
//!
//! The model is an intercept followed by smooth blocks; each block has one
//! penalty, an optional sum-to-zero row and its inequality rows.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};

use super::constraints::Constraints;
use super::pcls::{normal_equations, null_space, pcls, spd_inverse};
use crate::error::{GamrsError, Result};

/// One penalised smooth block of the coefficient vector.
#[derive(Debug, Clone)]
pub struct Block {
    /// First coefficient of the block (the intercept is coefficient 0).
    pub first: usize,
    /// Penalty on the block's coefficients, already rescaled as mgcv does.
    pub penalty: Array2<f64>,
}

impl Block {
    fn len(&self) -> usize {
        self.penalty.nrows()
    }
}

/// Everything the fit needs: model matrix (intercept first), blocks,
/// constraints and a strictly feasible start.
#[derive(Debug, Clone)]
pub struct ScasmModel {
    pub x: Array2<f64>,
    pub blocks: Vec<Block>,
    pub constraints: Constraints,
    pub beta0: Array1<f64>,
}

/// A 1-D shape-constrained smooth: covariate column, basis dimension and
/// monotone direction (`None` = unconstrained, still centred).
#[derive(Debug, Clone)]
pub struct ScSpec {
    pub col: usize,
    pub k: usize,
    pub shape: Option<super::constraints::Shape>,
}

/// A built model plus the bases needed to predict from it.
#[derive(Debug, Clone)]
pub struct ScasmDesign {
    pub model: ScasmModel,
    pub specs: Vec<ScSpec>,
    pub bases: Vec<super::bspline::BsBasis>,
}

impl ScasmDesign {
    /// `y ~ s(x_1, bs = "sc", ...) + ...` with mgcv's defaults
    /// (`m = c(3, 2)`, centred, penalty rescaled).
    pub fn build(data: &Array2<f64>, specs: &[ScSpec]) -> Result<Self> {
        use super::bspline::{column_means, scale_penalty, BsBasis};
        use super::constraints::{monotone_start, shape_rows};
        let n = data.nrows();
        let p = 1 + specs.iter().map(|s| s.k).sum::<usize>();
        let mut x = Array2::<f64>::zeros((n, p));
        x.column_mut(0).fill(1.0);
        let mut constraints = Constraints::empty(p);
        let mut beta0 = Array1::<f64>::zeros(p);
        let mut blocks = Vec::new();
        let mut bases = Vec::new();
        let mut first = 1;
        for spec in specs {
            let basis = BsBasis::from_data(data.column(spec.col), spec.k, (3, 2))?;
            let xb = basis.design(data.column(spec.col));
            let cm = column_means(&xb);
            let (penalty, _) = scale_penalty(&basis.penalty(), &xb);
            x.slice_mut(ndarray::s![.., first..first + spec.k])
                .assign(&xb);
            match spec.shape {
                Some(shape) => {
                    constraints.add_smooth(first, &shape_rows(shape, spec.k), Some(&cm));
                    beta0
                        .slice_mut(ndarray::s![first..first + spec.k])
                        .assign(&monotone_start(shape, &cm));
                }
                None => constraints.add_smooth(first, &Array2::zeros((0, spec.k)), Some(&cm)),
            }
            blocks.push(Block { first, penalty });
            bases.push(basis);
            first += spec.k;
        }
        Ok(Self {
            model: ScasmModel {
                x,
                blocks,
                constraints,
                beta0,
            },
            specs: specs.to_vec(),
            bases,
        })
    }

    /// Per-term contributions at new data (`predict(type = "terms")`): the
    /// basis at the new points times the term's coefficients. Columns
    /// follow `specs`.
    pub fn predict_terms(&self, beta: &Array1<f64>, data: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((data.nrows(), self.specs.len()));
        for (t, (spec, (basis, block))) in self
            .specs
            .iter()
            .zip(self.bases.iter().zip(&self.model.blocks))
            .enumerate()
        {
            let xb = basis.design(data.column(spec.col));
            let b = beta.slice(ndarray::s![block.first..block.first + spec.k]);
            out.column_mut(t).assign(&xb.dot(&b));
        }
        out
    }

    /// Standard errors of the term contributions from `vp`.
    pub fn predict_terms_se(&self, vp: &Array2<f64>, data: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((data.nrows(), self.specs.len()));
        for (t, (spec, (basis, block))) in self
            .specs
            .iter()
            .zip(self.bases.iter().zip(&self.model.blocks))
            .enumerate()
        {
            let xb = basis.design(data.column(spec.col));
            let r = block.first..block.first + spec.k;
            let v = vp.slice(ndarray::s![r.clone(), r]);
            let xv = xb.dot(&v);
            out.column_mut(t)
                .assign(&(&xv * &xb).sum_axis(Axis(1)).mapv(|s| s.max(0.0).sqrt()));
        }
        out
    }
}

/// State of one EFS iteration, for diagnostics and stage-by-stage tests.
#[derive(Debug, Clone)]
pub struct EfsIteration {
    pub lsp: Array1<f64>,
    pub active: Vec<usize>,
    pub edf_total: f64,
    pub scale: f64,
    pub tr_s: Array1<f64>,
    pub sedf: Array1<f64>,
    pub psp: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct ScasmFit {
    pub beta: Array1<f64>,
    pub sp: Array1<f64>,
    /// Effective degrees of freedom per coefficient (projected onto the
    /// active constraints' null space, as scasm reports them).
    pub edf: Array1<f64>,
    pub scale: f64,
    /// Bayesian posterior covariance, scale included, projected onto the
    /// null space of the sum-to-zero rows only (scasm's `Vp`).
    pub vp: Array2<f64>,
    /// Laplace approximate log restricted likelihood; scasm reports `-laml`
    /// as its REML score.
    pub laml: f64,
    pub active: Vec<usize>,
    pub fitted: Array1<f64>,
    pub iterations: Vec<EfsIteration>,
    pub converged: bool,
}

impl ScasmFit {
    pub fn reml(&self) -> f64 {
        -self.laml
    }
}

/// Coefficients at fixed smoothing parameters (Gaussian: one `pcls` solve,
/// started from the feasible point with an empty working set, as
/// `scasm.pirls` does for the identity link).
pub fn fit_fixed_sp(
    model: &ScasmModel,
    y: &Array1<f64>,
    sp: &Array1<f64>,
) -> Result<(Array1<f64>, Vec<usize>)> {
    let w = Array1::<f64>::ones(y.len());
    let pens: Vec<(usize, &Array2<f64>, f64)> = model
        .blocks
        .iter()
        .zip(sp.iter())
        .map(|(b, &l)| (b.first, &b.penalty, l))
        .collect();
    let (h, xty) = normal_equations(&model.x, &w, y, &pens);
    let c = &model.constraints;
    let fit = pcls(&h, &xty, &c.ain, &c.bin, &c.c, &model.beta0)?;
    Ok((fit.beta, fit.active))
}

/// mgcv's `initial.sp` (via `initial.spg` with unit Gaussian weights).
pub fn initial_sp(model: &ScasmModel) -> Array1<f64> {
    let x = &model.x;
    let p = x.ncols();
    let ldxx: Array1<f64> = x.map(|v| v * v).sum_axis(Axis(0));
    let mut ldss = Array1::<f64>::zeros(p);
    let mut pen = vec![false; p];
    let mut sp = Array1::<f64>::zeros(model.blocks.len());
    for (i, b) in model.blocks.iter().enumerate() {
        let s = &b.penalty;
        let k = b.len();
        let ma_s = s.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let thresh = f64::EPSILON.powf(0.8) * ma_s;
        let mut xx = Vec::new();
        let mut ss = Vec::new();
        for j in 0..k {
            let rs = s.row(j).iter().map(|v| v.abs()).sum::<f64>() / k as f64;
            let cs = s.column(j).iter().map(|v| v.abs()).sum::<f64>() / k as f64;
            let ds = s[[j, j]].abs();
            let on = rs > thresh && cs > thresh && ds > thresh;
            if on {
                ss.push(s[[j, j]]);
                xx.push(ldxx[b.first + j]);
            }
            pen[b.first + j] |= on;
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        sp[i] = mean(&xx) / mean(&ss);
        for j in 0..k {
            ldss[b.first + j] += sp[i] * s[[j, j]];
        }
    }
    let idx: Vec<usize> = (0..p)
        .filter(|&j| ldss[j] > 0.0 && pen[j] && ldxx[j] > 0.0)
        .collect();
    let ratio = |ldss: &Array1<f64>| {
        idx.iter()
            .map(|&j| ldxx[j] / (ldxx[j] + ldss[j]))
            .sum::<f64>()
            / idx.len() as f64
    };
    while ratio(&ldss) > 0.4 {
        sp *= 10.0;
        ldss *= 10.0;
    }
    while ratio(&ldss) < 0.4 {
        sp /= 10.0;
        ldss /= 10.0;
    }
    sp
}

/// Null-space basis of the sum-to-zero rows and the given active
/// inequality rows, built block by block (mgcv's `getZ`): identity on
/// unconstrained coefficients.
fn constraint_null_space(model: &ScasmModel, active: &[usize]) -> Result<Array2<f64>> {
    let p = model.x.ncols();
    let c = &model.constraints;
    let mut cols: Vec<Array2<f64>> = Vec::new();
    let mut covered = vec![false; p];
    let mut pieces: Vec<(usize, Array2<f64>)> = Vec::new();
    for b in &model.blocks {
        let range = b.first..b.first + b.len();
        let touches = |row: ndarray::ArrayView1<f64>| range.clone().any(|j| row[j] != 0.0);
        let mut rows: Vec<Array1<f64>> = Vec::new();
        for r in c.c.axis_iter(Axis(0)).filter(|r| touches(*r)) {
            rows.push(r.slice(ndarray::s![range.clone()]).to_owned());
        }
        for &a in active {
            let r = c.ain.row(a);
            if touches(r) {
                rows.push(r.slice(ndarray::s![range.clone()]).to_owned());
            }
        }
        let m = Array2::from_shape_fn((rows.len(), b.len()), |(i, j)| rows[i][j]);
        pieces.push((b.first, null_space(&m, b.len())?));
        range.for_each(|j| covered[j] = true);
    }
    for j in (0..p).filter(|j| !covered[*j]) {
        let mut e = Array2::<f64>::zeros((p, 1));
        e[[j, 0]] = 1.0;
        cols.push(e);
    }
    for (first, z) in pieces {
        let mut e = Array2::<f64>::zeros((p, z.ncols()));
        e.slice_mut(ndarray::s![first..first + z.nrows(), ..])
            .assign(&z);
        cols.push(e);
    }
    let views: Vec<_> = cols.iter().map(|a| a.view()).collect();
    ndarray::concatenate(Axis(1), &views).map_err(|e| GamrsError::Linalg(e.to_string()))
}

fn total_penalty(model: &ScasmModel, sp: &Array1<f64>) -> Array2<f64> {
    let p = model.x.ncols();
    let mut st = Array2::<f64>::zeros((p, p));
    for (b, &l) in model.blocks.iter().zip(sp.iter()) {
        let k = b.len();
        st.slice_mut(ndarray::s![b.first..b.first + k, b.first..b.first + k])
            .scaled_add(l, &b.penalty);
    }
    st
}

/// Inverse of a symmetric matrix, or the pseudo-inverse of its positive
/// part when it is not positive definite (mgcv's `c2inv`).
fn c2inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    if let Ok(inv) = spd_inverse(a) {
        return Ok(inv);
    }
    let (ev, v) = a
        .eigh(UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(e.to_string()))?;
    let top = ev.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let inv_ev = ev.mapv(|x| {
        if x > top * f64::EPSILON.powf(0.9) {
            1.0 / x
        } else {
            0.0
        }
    });
    Ok((&v * &inv_ev.insert_axis(Axis(0))).dot(&v.t()))
}

/// Numerical rank of a symmetric positive semi-definite matrix, with the
/// tolerance LAPACK's pivoted Cholesky uses (`n ε max|a_jj|`).
fn psd_rank(a: &Array2<f64>) -> Result<usize> {
    let (ev, _) = a
        .eigh(UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(e.to_string()))?;
    let dmax = a.diag().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = a.nrows() as f64 * f64::EPSILON * dmax;
    Ok(ev.iter().filter(|&&x| x > tol).count())
}

/// Log pseudo-determinant (sum of logs of the non-zero eigenvalues) of a
/// positive semi-definite matrix (mgcv's `lpdgDet`).
fn log_pdet(a: &Array2<f64>) -> Result<(f64, usize)> {
    let r = psd_rank(a)?;
    let (ev, _) = a
        .eigh(UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(e.to_string()))?;
    let mut ev = ev.to_vec();
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
    Ok((ev[..r].iter().map(|v| v.ln()).sum(), r))
}

struct Projected {
    vb: Array2<f64>,
    edf: Array1<f64>,
    sedf: Array1<f64>,
    psp: Array1<f64>,
}

/// mgcv's `getVb(project = TRUE)` for a Gaussian fit (H = XᵀX is positive
/// semi-definite, so no indefinite-Hessian repair is needed).
fn project(
    model: &ScasmModel,
    h: &Array2<f64>,
    beta: &Array1<f64>,
    sp: &Array1<f64>,
    z: &Array2<f64>,
) -> Result<Projected> {
    let st = total_penalty(model, sp);
    let ht = z.t().dot(h).dot(z);
    let hp = &ht + &z.t().dot(&st).dot(z);
    let vb = z.dot(&c2inv(&hp)?).dot(&z.t());
    let hproj = z.dot(&ht).dot(&z.t());
    let edf = (&vb * &hproj).sum_axis(Axis(1));
    let mut sedf = Array1::<f64>::zeros(model.blocks.len());
    let mut psp = Array1::<f64>::zeros(model.blocks.len());
    for (i, b) in model.blocks.iter().enumerate() {
        let r = b.first..b.first + b.len();
        let vbb = vb.slice(ndarray::s![r.clone(), r.clone()]);
        sedf[i] = (&vbb * &b.penalty).sum();
        let bb = beta.slice(ndarray::s![r]);
        psp[i] = bb.dot(&b.penalty.dot(&bb)).max(f64::MIN_POSITIVE);
    }
    Ok(Projected { vb, edf, sedf, psp })
}

/// `trSiSZ` for single-penalty blocks: rank(ZᵀS_jZ) / λ_j, with Z the
/// block's own constraint null space.
fn tr_s_inverse(model: &ScasmModel, z: &Array2<f64>, sp: &Array1<f64>) -> Result<Array1<f64>> {
    let mut out = Array1::<f64>::zeros(model.blocks.len());
    for (i, b) in model.blocks.iter().enumerate() {
        let r = b.first..b.first + b.len();
        let zb = z.slice(ndarray::s![r, ..]);
        // Columns of Z that belong to this block.
        let keep: Vec<usize> = (0..zb.ncols())
            .filter(|&j| zb.column(j).iter().any(|v| *v != 0.0))
            .collect();
        let zk = Array2::from_shape_fn((zb.nrows(), keep.len()), |(a, j)| zb[[a, keep[j]]]);
        let zsz = zk.t().dot(&b.penalty).dot(&zk);
        out[i] = psd_rank(&zsz)? as f64 / sp[i];
    }
    Ok(out)
}

/// Fit by scasm's EFS loop (`scasm.fit`, Gaussian branch).
pub fn fit_gaussian(model: &ScasmModel, y: &Array1<f64>) -> Result<ScasmFit> {
    let n = y.len();
    let m = model.blocks.len();
    let h = model.x.t().dot(&model.x);
    let mut lsp = initial_sp(model).mapv(f64::ln);
    let mut dlsp = Array1::<f64>::zeros(m);
    let mut alpha = Array1::<f64>::ones(m);
    let mut same_sign_prev = vec![false; m];
    let mut psp_old = Array1::<f64>::zeros(m);
    let mut eps = 0.01_f64;
    let mut iterations = Vec::new();
    let mut converged = false;
    let (mut beta, mut active, mut proj, mut scale);
    let mut i = 1;
    loop {
        let sp = lsp.mapv(f64::exp);
        let (b, a) = fit_fixed_sp(model, y, &sp)?;
        beta = b;
        active = a;
        let z = constraint_null_space(model, &active)?;
        proj = project(model, &h, &beta, &sp, &z)?;
        let resid = y - &model.x.dot(&beta);
        scale = resid.dot(&resid) / (n as f64 - proj.edf.sum());
        if m == 0 {
            converged = true;
            break;
        }
        let tr_s = tr_s_inverse(model, &z, &sp)?;
        iterations.push(EfsIteration {
            lsp: lsp.clone(),
            active: active.clone(),
            edf_total: proj.edf.sum(),
            scale,
            tr_s: tr_s.clone(),
            sedf: proj.sedf.clone(),
            psp: proj.psp.clone(),
        });
        let mut redf = (&tr_s - &proj.sedf) * &sp;
        if redf.iter().any(|&v| v <= 0.0) {
            redf.mapv_inplace(|v| if v < 1e-7 { 1e-7 } else { v });
        }
        let sedf = &proj.sedf * &sp;
        let dlsp0 = dlsp.clone();
        dlsp = Array1::from_shape_fn(m, |j| {
            scale.ln() + redf[j].ln() - (proj.psp[j] * sp[j]).ln()
        });
        let maxstep = dlsp.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if i == 1 && maxstep > 8.0 {
            dlsp *= 8.0 / maxstep;
        }
        if i > 2 && maxstep > 4.0 {
            dlsp *= 4.0 / maxstep;
        }
        for j in 0..m {
            if (redf[j] < 0.05 && dlsp[j] > 0.0) || (sedf[j] < 0.05 && dlsp[j] < 0.0) {
                dlsp[j] = 0.0;
            }
        }
        if i > 1 {
            let same: Vec<bool> = (0..m).map(|j| dlsp[j] * dlsp0[j] > 0.0).collect();
            for j in 0..m {
                if !same[j] {
                    alpha[j] /= 2.0;
                }
            }
            if i > 2 {
                for j in 0..m {
                    if same[j] && same_sign_prev[j] {
                        alpha[j] = (alpha[j] * 1.2).min(2.0);
                    }
                }
            }
            same_sign_prev = same;
        }
        let lsp1 = &dlsp * &alpha + &lsp;
        if i == 7 {
            eps = 1e-3;
        }
        let psp_change = (0..m).fold(0.0_f64, |a, j| {
            a.max((psp_old[j] - proj.psp[j]).abs() / proj.psp[j].max(1e-4))
        });
        let step = (0..m).fold(0.0_f64, |a, j| a.max((lsp1[j] - lsp[j]).abs()));
        if (i > 1 && psp_change < 0.01) || step < 0.01 {
            if eps < 0.005 {
                converged = true;
                break;
            }
            eps = 1e-3;
        }
        lsp = lsp1;
        if i == 200 {
            break;
        }
        psp_old = proj.psp.clone();
        i += 1;
    }

    // Final covariance and REML: projected onto the sum-to-zero rows only.
    let sp = lsp.mapv(f64::exp);
    let z0 = constraint_null_space(model, &[])?;
    let vp = project(model, &h, &beta, &sp, &z0)?.vb * scale;
    let st = total_penalty(model, &sp);
    let st_t = z0.t().dot(&st).dot(&z0);
    let hp_t = z0.t().dot(&h).dot(&z0) + &st_t;
    let (ldet_s, rank_s) = log_pdet(&st_t)?;
    let (ldet_h, _) = log_pdet(&hp_t)?;
    let fitted = model.x.dot(&beta);
    let resid = y - &fitted;
    let pdev = resid.dot(&resid) + beta.dot(&st.dot(&beta));
    let mdim = (hp_t.ncols() - rank_s) as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let laml = (-pdev / scale + ldet_s - ldet_h) / 2.0 + mdim * two_pi.ln() / 2.0
        - n as f64 * (two_pi * scale).ln() / 2.0;
    Ok(ScasmFit {
        beta,
        sp,
        edf: proj.edf,
        scale,
        vp,
        laml,
        active,
        fitted,
        iterations,
        converged,
    })
}
