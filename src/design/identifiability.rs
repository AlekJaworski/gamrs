//! Coefficients the data and the penalties cannot pin down.
//!
//! A direction `v` with `X v = 0` and `S_j v = 0` for every `j` leaves
//! `A = X'WX + Σ λ_j S_j` singular at every λ, so the fit fails in Cholesky
//! however the outer search moves. Two identical 0/1 columns do it, and so do
//! two count columns that always sum to a constant (together they reproduce
//! the intercept). mgcv fits these by dropping the aliased coefficients in its
//! pivoted QR (`gdi.c`, `rank.tol`) and reporting them as exactly 0; this does
//! the same, once, before the fit.
//!
//! Only the joint penalty null space is searched: a direction any penalty
//! touches is identifiable for every finite λ, and that null space is the one
//! the REML score already treats as unpenalised.

use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::{Cholesky, Eigh, UPLO};

use crate::error::{GamrsError, Result};

use super::{rank_and_log_pseudo_det, PreparedDesign};

/// Eigenvalue cut on the unit-diagonal Gram of the unpenalised directions.
/// A Gram squares the singular values, so mgcv's `sqrt(eps)` on R becomes
/// `eps` here — which is the round-off floor of forming the Gram. 1e-12 sits
/// clear of that floor (an exact copy lands near 1e-16) while still only
/// catching columns that agree to about six significant figures.
const GRAM_TOL: f64 = 1e-12;

/// Same relative cut `rank_and_log_pseudo_det` uses to call a penalty
/// eigenvalue zero, so "unpenalised" means what the score means by it.
const PENALTY_TOL: f64 = 1e-10;

/// A dropped column must carry at least this much of the aliased direction
/// that is not already removed, or zeroing it would leave the rest near-singular.
const PIVOT_TOL: f64 = 1e-2;

/// Columns of `x_design` to drop so that every remaining coefficient is
/// identifiable, in increasing order. Empty for a well-posed design.
///
/// Prefers the latest unpenalised column in each aliased set (mgcv keeps
/// `water` and zeroes the later `has_docks`), and the intercept last; a
/// penalised column is only dropped when no unpenalised one can absorb the
/// direction, e.g. two smooths of the same covariate.
pub(crate) fn aliased_columns(
    x_design: &Array2<f64>,
    s_list: &[Array2<f64>],
    prior_weights: Option<ArrayView1<f64>>,
) -> Result<Vec<usize>> {
    let p = x_design.ncols();
    if p == 0 {
        return Ok(Vec::new());
    }
    let mut wx = x_design.clone();
    if let Some(w) = prior_weights {
        for (mut row, &wi) in wx.axis_iter_mut(Axis(0)).zip(w.iter()) {
            let s = wi.max(0.0).sqrt();
            row.mapv_inplace(|v| v * s);
        }
    }
    let mut gram = wx.t().dot(&wx);

    // Unit-diagonal scaling makes every tolerance below scale-free: a column
    // in dollars and a 0/1 tag are judged alike.
    let scale: Array1<f64> = gram
        .diag()
        .mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 });
    for i in 0..p {
        for j in 0..p {
            gram[[i, j]] *= scale[i] * scale[j];
        }
    }
    let mut penalty = Array2::<f64>::zeros((p, p));
    for s_j in s_list {
        let norm = s_j.diag().iter().cloned().fold(0.0_f64, f64::max);
        if norm > 0.0 {
            penalty.scaled_add(1.0 / norm, s_j);
        }
    }
    for i in 0..p {
        for j in 0..p {
            penalty[[i, j]] *= scale[i] * scale[j];
        }
    }
    let unpenalised: Vec<bool> = (0..p)
        .map(|i| penalty.row(i).iter().all(|&v| v == 0.0))
        .collect();

    // Fast path: a comfortably positive-definite `gram + penalty` has no
    // direction both miss.
    if let Ok(l) = (&gram + &penalty).cholesky(UPLO::Lower) {
        if l.diag().iter().all(|&d| d * d > 1e-8) {
            return Ok(Vec::new());
        }
    }

    let null_basis = if s_list.is_empty() {
        Array2::<f64>::eye(p)
    } else {
        let (eigs, vecs) = penalty
            .eigh(UPLO::Lower)
            .map_err(|e| GamrsError::Linalg(format!("identifiability check: {e}")))?;
        let tol = eigs.iter().cloned().fold(0.0_f64, f64::max).max(1.0) * PENALTY_TOL;
        let cols: Vec<usize> = (0..p).filter(|&i| eigs[i] <= tol).collect();
        vecs.select(Axis(1), &cols)
    };
    if null_basis.ncols() == 0 {
        return Ok(Vec::new());
    }
    let null_gram = null_basis.t().dot(&gram).dot(&null_basis);
    let (mu, w) = null_gram
        .eigh(UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(format!("identifiability check: {e}")))?;
    let tol = mu.iter().cloned().fold(0.0_f64, f64::max).max(1.0) * GRAM_TOL;
    let bad: Vec<usize> = (0..mu.len()).filter(|&i| mu[i] <= tol).collect();
    if bad.is_empty() {
        return Ok(Vec::new());
    }
    // Rows of `aliased` are the columns' shares of the aliased directions.
    let aliased = null_basis.dot(&w.select(Axis(1), &bad));
    let d = bad.len();

    let order = (0..p)
        .rev()
        .filter(|&i| unpenalised[i])
        .chain((0..p).rev().filter(|&i| !unpenalised[i]));
    // Greedy row selection: accept a column when its row is not already
    // spanned by the rows chosen so far (Gram-Schmidt on the residual).
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(d);
    let mut dropped = Vec::with_capacity(d);
    for i in order {
        if dropped.len() == d {
            break;
        }
        let mut r = aliased.row(i).to_owned();
        for q in &basis {
            let c = r.dot(q);
            r.scaled_add(-c, q);
        }
        let norm = r.dot(&r).sqrt();
        if norm > PIVOT_TOL {
            basis.push(r / norm);
            dropped.push(i);
        }
    }
    if dropped.len() < d {
        return Err(GamrsError::SingularSystem(format!(
            "{d} coefficient direction(s) are not identifiable and no set of columns removes them"
        )));
    }
    dropped.sort_unstable();
    Ok(dropped)
}

/// `prep` without the `dropped` columns. A penalty that loses a column has its
/// rank and log pseudo-determinant recomputed; an unpenalised column leaves
/// every penalty unchanged.
pub(crate) fn without_columns(prep: PreparedDesign, dropped: &[usize]) -> Result<PreparedDesign> {
    let keep: Vec<usize> = kept(prep.x_design.ncols(), dropped).collect();
    let mut rank_s_list = prep.rank_s_list;
    let mut log_pseudo_det_s_list = prep.log_pseudo_det_s_list;
    let mut s_list = Vec::with_capacity(prep.s_list.len());
    for (j, s_j) in prep.s_list.iter().enumerate() {
        let reduced = s_j
            .select(Axis(0), &keep)
            .select(Axis(1), &keep)
            .as_standard_layout()
            .into_owned();
        let touched = dropped
            .iter()
            .any(|&i| s_j.row(i).iter().any(|&v| v != 0.0));
        if touched {
            let (rank, log_det) = rank_and_log_pseudo_det(reduced.view())?;
            rank_s_list[j] = rank;
            log_pseudo_det_s_list[j] = log_det;
        }
        s_list.push(reduced);
    }
    let mp = keep.len().saturating_sub(rank_s_list.iter().sum::<usize>());
    Ok(PreparedDesign {
        x_design: prep
            .x_design
            .select(Axis(1), &keep)
            .as_standard_layout()
            .into_owned(),
        s_list,
        rank_s_list,
        log_pseudo_det_s_list,
        mp,
        predictor: prep.predictor,
    })
}

/// Re-insert dropped coefficients as exact zeros.
pub(crate) fn with_zero_coefficients(beta: ArrayView1<f64>, dropped: &[usize]) -> Array1<f64> {
    let p = beta.len() + dropped.len();
    let mut full = Array1::<f64>::zeros(p);
    for (a, i) in kept(p, dropped).enumerate() {
        full[i] = beta[a];
    }
    full
}

/// Re-insert dropped coefficients' rows and columns as zero variance.
pub(crate) fn with_zero_variance(vcov: &Array2<f64>, dropped: &[usize]) -> Array2<f64> {
    let p = vcov.nrows() + dropped.len();
    let mut full = Array2::<f64>::zeros((p, p));
    for (a, i) in kept(p, dropped).enumerate() {
        for (b, j) in kept(p, dropped).enumerate() {
            full[[i, j]] = vcov[[a, b]];
        }
    }
    full
}

fn kept(p: usize, dropped: &[usize]) -> impl Iterator<Item = usize> + '_ {
    (0..p).filter(move |i| dropped.binary_search(i).is_err())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn design(cols: &[Vec<f64>]) -> Array2<f64> {
        let n = cols[0].len();
        Array2::from_shape_fn((n, cols.len()), |(i, j)| cols[j][i])
    }

    #[test]
    fn well_posed_design_drops_nothing() {
        let x = design(&[
            vec![1.0; 4],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 2.0, 4.0, 3.0],
        ]);
        assert!(aliased_columns(&x, &[], None).unwrap().is_empty());
    }

    #[test]
    fn exact_copy_drops_the_later_column() {
        let tag = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let x = design(&[
            vec![1.0; 6],
            tag.clone(),
            vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
            tag,
        ]);
        assert_eq!(aliased_columns(&x, &[], None).unwrap(), vec![3]);
    }

    #[test]
    fn columns_summing_to_a_constant_drop_the_later_one_not_the_intercept() {
        // baths_full + baths_half == 3 on every row.
        let full = vec![2.0, 2.0, 1.0, 2.0, 2.0];
        let half: Vec<f64> = full.iter().map(|f| 3.0 - f).collect();
        let x = design(&[vec![1.0; 5], full, half]);
        assert_eq!(aliased_columns(&x, &[], None).unwrap(), vec![2]);
    }

    #[test]
    fn rescaled_copy_is_caught() {
        let gla = vec![1200.0, 1500.0, 900.0, 2100.0, 1750.0];
        let sqm: Vec<f64> = gla.iter().map(|v| v * 0.092903).collect();
        let x = design(&[vec![1.0; 5], gla, sqm]);
        assert_eq!(aliased_columns(&x, &[], None).unwrap(), vec![2]);
    }

    #[test]
    fn penalised_columns_are_identifiable_through_the_penalty() {
        // Two columns equal on the data, but a penalty separates them.
        let c = vec![0.0, 1.0, 2.0, 1.0];
        let x = design(&[vec![1.0; 4], c.clone(), c]);
        let s = array![[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0]];
        assert!(aliased_columns(&x, &[s], None).unwrap().is_empty());
    }

    #[test]
    fn zero_weight_rows_do_not_identify() {
        // The two columns differ only on a row with prior weight 0.
        let x = design(&[
            vec![1.0; 4],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ]);
        let w = array![1.0, 1.0, 1.0, 0.0];
        assert!(aliased_columns(&x, &[], None).unwrap().is_empty());
        assert_eq!(aliased_columns(&x, &[], Some(w.view())).unwrap(), vec![2]);
    }
}
