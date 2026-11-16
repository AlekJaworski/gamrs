//! Penalised constrained least squares: a port of mgcv's `pcls` (`PCLS` /
//! `QPCLS` in `qp.c`).
//!
//! Solves `min ‖W^{1/2}(y − Xβ)‖² + βᵀ B β` subject to `A β ≥ 0`, from a
//! feasible `β₀`, by the primal active-set method of
//! Gill, Murray & Wright (1981, §5.3.3) exactly as `QPCLS` runs it: minimise
//! in the null space of the working set, step to the nearest blocking
//! constraint (lowest row index on ties) and add it, or at a working-set
//! minimum drop the inequality with the most negative Lagrange multiplier.
//! A constraint deleted more than 100 times is pinned in the working set.
//!
//! `QPCLS` updates QR factors of the stacked `[W^{1/2}X; √B]` by Givens
//! rotations as the working set changes. Here each working-set solve is
//! recomputed from `H = XᵀWX + B` restricted to an orthonormal null-space
//! basis of the working constraints. The iterates, the working set and the
//! solution are the same (the problem is strictly convex on the constraint
//! null space, so the minimiser is unique); only the cost per step differs,
//! which is immaterial at the basis sizes smooths have.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SolveC;

use crate::error::{GamrsError, Result};

/// Result of a constrained solve.
#[derive(Debug, Clone)]
pub struct PclsFit {
    pub beta: Array1<f64>,
    /// Inequality rows in the final working set, in the order they entered
    /// (0-based; mgcv's `attr(p, "active")` is these plus one).
    pub active: Vec<usize>,
}

const MAX_DELETIONS: usize = 100;
const MAX_STEPS: usize = 10_000;

/// Solve with Hessian `h = XᵀWX + B` and `xty = XᵀWy`, inequality rows
/// `ain β ≥ 0`, from feasible `beta0`. Identifiability constraints are
/// absorbed into the basis before this point, so there are no equality rows.
pub(crate) fn pcls(
    h: &Array2<f64>,
    xty: &Array1<f64>,
    ain: &Array2<f64>,
    beta0: &Array1<f64>,
) -> Result<PclsFit> {
    let p = beta0.len();
    let mut beta = beta0.clone();
    let mut working: Vec<usize> = Vec::new();
    let mut pinned: Vec<bool> = Vec::new();
    let mut in_set = vec![false; ain.nrows()];
    let mut deletions = vec![0usize; ain.nrows()];

    for _ in 0..MAX_STEPS {
        let m = working_rows(ain, &working);
        let z = null_space(&m, p)?;
        if z.ncols() == 0 {
            return Err(GamrsError::SingularSystem(
                "pcls: working set fixes every coefficient".into(),
            ));
        }
        // Minimise in the null space: (ZᵀHZ) pz = Zᵀ(XᵀWy − Hβ).
        let grad_neg = xty - &h.dot(&beta);
        let zhz = z.t().dot(h).dot(&z);
        let rhs = z.t().dot(&grad_neg);
        let pz = zhz.solvec(&rhs).map_err(|e| {
            GamrsError::SingularSystem(format!("pcls: rank deficiency in model ({e})"))
        })?;
        let pk = z.dot(&pz);

        match step_to_constraint(ain, &in_set, &beta, &pk, &z) {
            (next, Some(row)) => {
                beta = next;
                working.push(row);
                in_set[row] = true;
                pinned.push(deletions[row] > MAX_DELETIONS);
            }
            (next, None) => {
                beta = next;
                match most_negative_multiplier(h, xty, ain, &working, &pinned, &beta)? {
                    Some(pos) => {
                        let row = working.remove(pos);
                        pinned.remove(pos);
                        in_set[row] = false;
                        deletions[row] += 1;
                    }
                    None => {
                        return Ok(PclsFit {
                            beta,
                            active: working,
                        })
                    }
                }
            }
        }
    }
    Err(GamrsError::NotConverged {
        iters: MAX_STEPS,
        grad_norm: f64::NAN,
    })
}

/// `LSQPstep`: step from `beta` along `pk`, stopping at the first inequality
/// outside the working set that the full step would violate.
///
/// A row in the span of the working rows cannot block: it is a combination
/// of rows held at zero, so it is zero now and stays zero along `pk`, which
/// lies in their null space `z`. In exact arithmetic `a·pk` is then exactly
/// zero and QPCLS's `|a·pk| > 0` test skips it; in floating point it is
/// rounding noise, and stepping to it would add a dependent row to the
/// working set and wreck the null-space basis. So such rows — those whose
/// component in the null space, `‖zᵀa‖`, is at rounding level relative to
/// `‖a‖` — are skipped explicitly. The shape rows of a tensor are linearly
/// dependent (differences along both margins around a cell sum to zero), so
/// this happens whenever a whole patch of the surface is flat.
fn step_to_constraint(
    ain: &Array2<f64>,
    in_set: &[bool],
    beta: &Array1<f64>,
    pk: &Array1<f64>,
    z: &Array2<f64>,
) -> (Array1<f64>, Option<usize>) {
    let full = beta + pk;
    let mut next = full.clone();
    let mut alpha_min = 1.0;
    let mut hit = None;
    for i in 0..ain.nrows() {
        if in_set[i] {
            continue;
        }
        let a = ain.row(i);
        if -a.dot(&full) > 0.0 && !in_span(a, z) {
            let apk = a.dot(pk);
            if apk.abs() > 0.0 {
                let alpha = -a.dot(beta) / apk;
                if alpha < alpha_min {
                    alpha_min = alpha.max(0.0);
                    hit = Some(i);
                    next = beta + &(pk * alpha_min);
                }
            }
        }
    }
    (next, hit)
}

/// `LSQPlagrange`: multipliers of the working set from `Mᵀ l = g`,
/// `g = Hβ − XᵀWy`; returns the working-set position of the inequality
/// with the most negative multiplier, skipping pinned ones.
fn most_negative_multiplier(
    h: &Array2<f64>,
    xty: &Array1<f64>,
    ain: &Array2<f64>,
    working: &[usize],
    pinned: &[bool],
    beta: &Array1<f64>,
) -> Result<Option<usize>> {
    if working.is_empty() {
        return Ok(None);
    }
    let m = working_rows(ain, working);
    let g = h.dot(beta) - xty;
    let mmt = m.dot(&m.t());
    let l = mmt.solvec(&m.dot(&g)).map_err(|e| {
        GamrsError::SingularSystem(format!("pcls: dependent working constraints ({e})"))
    })?;
    let mut best = None;
    let mut most = 0.0;
    for (pos, &pin) in pinned.iter().enumerate() {
        if !pin && l[pos] < most {
            most = l[pos];
            best = Some(pos);
        }
    }
    Ok(best)
}

/// Whether row `a` lies (to rounding) in the span of the working rows,
/// i.e. has no component in their null space `z`.
fn in_span(a: ndarray::ArrayView1<f64>, z: &Array2<f64>) -> bool {
    let along = z.t().dot(&a);
    along.dot(&along).sqrt() <= SPAN_TOL * a.dot(&a).sqrt()
}

/// Relative size of a row's null-space component below which it counts as
/// a combination of the working rows: a few hundred ulps.
const SPAN_TOL: f64 = 1e-12;

fn working_rows(ain: &Array2<f64>, working: &[usize]) -> Array2<f64> {
    ain.select(Axis(0), working)
}

/// Orthonormal basis (p × (p − r)) of the null space of the r × p matrix
/// `m` (full row rank), from a complete Householder QR of `mᵀ`.
pub(crate) fn null_space(m: &Array2<f64>, p: usize) -> Result<Array2<f64>> {
    let r = m.nrows();
    if r == 0 {
        return Ok(Array2::eye(p));
    }
    let q = householder_q(&m.t().to_owned())?;
    Ok(q.slice(ndarray::s![.., r..]).to_owned())
}

/// Full orthogonal Q (p × p) of the QR factorisation of `a` (p × r, r ≤ p).
fn householder_q(a: &Array2<f64>) -> Result<Array2<f64>> {
    let (p, r) = a.dim();
    let mut a = a.clone();
    let mut q = Array2::<f64>::eye(p);
    for j in 0..r {
        let x = a.slice(ndarray::s![j.., j]).to_owned();
        let norm = x.dot(&x).sqrt();
        if norm == 0.0 {
            return Err(GamrsError::SingularSystem(
                "pcls: dependent constraints".into(),
            ));
        }
        let mut v = x;
        let alpha = if v[0] > 0.0 { -norm } else { norm };
        v[0] -= alpha;
        let vnorm2 = v.dot(&v);
        if vnorm2 == 0.0 {
            continue;
        }
        // A ← (I − 2vvᵀ/vᵀv) A on rows j.., Q ← Q (I − 2vvᵀ/vᵀv) on cols j..
        let mut sub = a.slice_mut(ndarray::s![j.., j..]);
        let proj = v.dot(&sub) * (2.0 / vnorm2);
        for (i, vi) in v.iter().enumerate() {
            sub.row_mut(i).scaled_add(-vi, &proj);
        }
        let mut qs = q.slice_mut(ndarray::s![.., j..]);
        let qv = qs.dot(&v) * (2.0 / vnorm2);
        for (i, mut row) in qs.axis_iter_mut(Axis(0)).enumerate() {
            row.scaled_add(-qv[i], &v);
        }
    }
    Ok(q)
}
