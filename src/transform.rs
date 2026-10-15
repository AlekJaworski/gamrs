//! Basis-layer transforms — sum-to-zero centering + stable reparam.
//!
//! Two `BasisTransform` impls live here, both following the same recipe:
//! constraint matrix `C` of shape `(k_inner, k_self)` ⇒ rotated basis
//! `B_self = B_inner · C`, rotated penalty `S_self = C' · S · C`.
//!
//! - **`SumToZero<B>`** — mgcv's default identifiability constraint for
//!   univariate smooths: enforce `1' · X · β = 0` so the smooth has mean
//!   zero across the *fit* rows, with the intercept absorbing the level.
//!   `C` is `(k, k-1)` whose columns span the null-space of `t = colSums(X_fit)`,
//!   built via a Householder reflection. Mirrors `mgcv::nat.param` /
//!   `mgcv::smoothCon`'s default centering path.
//!
//!   The centering row is one *choice* of constraint, not the only one:
//!   [`SumToZero::from_point_constraint`] swaps it for mgcv's `s(x, pc=v)`
//!   — the smooth is pinned to zero at `x = v` instead of on average.
//!   Absorption, dimension (`k-1`) and penalty rotation are identical
//!   either way, which is why both live on this one type.
//!
//! - **`StableReparam<B>`** — mgcv's `Sl.initial.repara` analog. Rotates
//!   `B` so the inner penalty becomes diagonal (eigenvalue-ordered, range
//!   space first). `C := V`, the eigenvector matrix of `S_inner`. The
//!   diagonal-penalty structure dramatically reduces the condition number
//!   of the inner system `A = X'WX + λS` on ill-conditioned smooths,
//!   closing the residual rel-err gap on the `low_signal_n1000_k10`
//!   fixture (architecture-assumptions.md §C4-note).
//!
//! Composition is **type-level**: `StableReparam<SumToZero<CrSpline>>` is
//! a distinct type from `SumToZero<CrSpline>` and the compiler enforces
//! that the inner-solve, score, and vcov-rebuild all see the rotated
//! basis. Runtime flags for what's structurally a basis change are
//! explicitly forbidden by architecture-assumptions.md §B2.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{Eigh, UPLO};

use crate::error::{GamrsError, Result};
use crate::traits::{Basis, BasisTransform};

pub struct SumToZero<B: Basis> {
    inner: B,
    /// `(k_inner, k_inner - 1)` constraint matrix.
    c: Array2<f64>,
}

impl<B: Basis> SumToZero<B> {
    /// Build the constraint from the fit-time design matrix `X_fit = inner.evaluate(x_fit)`.
    /// The constraint is data-dependent because mgcv computes it from the
    /// *training* column sums (not the basis-vs-uniform integral).
    pub fn from_fit_design(inner: B, x_fit_design: ArrayView2<f64>) -> Self {
        let t = x_fit_design.sum_axis(Axis(0));
        Self::from_constraint_row(inner, t.view())
    }

    /// mgcv's point constraint `s(x, pc = point)`: the smooth passes
    /// through zero at `point`, and the intercept takes the level the
    /// centering constraint would have taken.
    ///
    /// The constraint row is the *uncentred* basis evaluated at the point
    /// — `mgcv:::smooth.construct3` sets `object$C <- Predict.matrix(object,
    /// pc)$X` and lets the ordinary absorption path handle the rest, so
    /// the smooth still spans `k-1` columns. `point` is a single row of
    /// covariate values, shape `(1, input_dim)`.
    ///
    /// Same model space as [`Self::from_fit_design`], so fitted values,
    /// λ̂ and edf are invariant: the CR penalty annihilates constants, and
    /// a constraint row only decides how a fitted function splits between
    /// the intercept and the smooth.
    pub fn from_point_constraint(inner: B, point: ArrayView2<f64>) -> Self {
        let t = inner.evaluate(point);
        Self::from_constraint_row(inner, t.row(0))
    }

    /// Absorb the single linear constraint `t' · β = 0` into the basis.
    fn from_constraint_row(inner: B, t: ArrayView1<f64>) -> Self {
        let c = nullspace_householder(t);
        Self { inner, c }
    }
}

impl<B: Basis> BasisTransform for SumToZero<B> {
    type Inner = B;
    fn inner(&self) -> &B {
        &self.inner
    }
    fn matrix(&self) -> ArrayView2<'_, f64> {
        self.c.view()
    }
}

impl<B: Basis> Basis for SumToZero<B> {
    fn dim(&self) -> usize {
        self.c.ncols()
    }
    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }
    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.inner.evaluate(x).dot(&self.c)
    }
    fn d1(&self, x: ArrayView2<f64>, axis: usize) -> Array2<f64> {
        self.inner.d1(x, axis).dot(&self.c)
    }
    fn penalties(&self) -> Vec<Array2<f64>> {
        self.inner
            .penalties()
            .into_iter()
            .map(|s| self.c.t().dot(&s).dot(&self.c))
            .collect()
    }
}

// =============================================================================
// StableReparam — mgcv `Sl.initial.repara` analog.
// =============================================================================

/// Stable reparameterisation (mgcv `Sl.initial.repara` analog) — rotates
/// the basis so the inner penalty becomes diagonal (eigenvalues on the
/// diagonal, range space first, null space last).
///
/// Construction:
///
/// ```text
///   S = V · diag(λ_1 ≥ … ≥ λ_k) · V'    (eigendecomposition of the inner penalty)
///   B_new = B_inner · V                  (rotate the design)
///   S_new = V' · S_inner · V = diag(λ)   (rotated penalty)
/// ```
///
/// The diagonal structure of `S_new` dramatically lowers the condition
/// number of `A = X'WX + λS`, which is the suspected root cause of the
/// §C4-note residual on `low_signal_n1000_k10` (rel-err ~2.3e-6,
/// otherwise unrecoverable from the Cholesky-vs-LU path).
///
/// Typical composition: `StableReparam<SumToZero<CrSpline>>` — the
/// reparam sits **outside** the centering, because the relevant penalty
/// for the inner solve is the centred-block penalty `S_centred = C' S_raw C`.
/// Build via [`StableReparam::from_inner_penalty`] after centring.
///
/// Predictions are basis-invariant: if β̂_stable is the StableReparam fit
/// and β̂_unrot is the no-rotation fit, then `X · V · β̂_stable = X · β̂_unrot`
/// to FP — only the coefficient representation differs.
pub struct StableReparam<B: Basis> {
    inner: B,
    /// Orthogonal rotation matrix `V`, shape `(k_inner, k_inner)`.
    /// Columns are eigenvectors of the inner penalty in descending
    /// eigenvalue order (range space first).
    v: Array2<f64>,
    /// Eigenvalues of the inner penalty in descending order (length k_inner).
    /// Cached so consumers don't redo the eigendecomposition.
    eigvals: Array1<f64>,
}

impl<B: Basis> StableReparam<B> {
    /// Build the reparam from the inner basis's penalty. For a single-smooth
    /// model the caller typically passes `inner.penalties().pop().unwrap()`.
    ///
    /// `inner_penalty` must be square, symmetric, and have side length
    /// `inner.dim()`. The eigendecomposition uses LAPACK `dsyevd` via
    /// `ndarray-linalg::Eigh` on the lower triangle.
    pub fn from_inner_penalty(inner: B, inner_penalty: ArrayView2<f64>) -> Result<Self> {
        let k = inner.dim();
        if inner_penalty.nrows() != k || inner_penalty.ncols() != k {
            return Err(GamrsError::InvalidParameter(format!(
                "StableReparam: inner_penalty must be ({k}, {k}); got ({}, {})",
                inner_penalty.nrows(),
                inner_penalty.ncols()
            )));
        }
        let (eigvals, v) = descending_eigh(inner_penalty)?;
        Ok(Self { inner, v, eigvals })
    }

    /// Eigenvalues of the inner penalty (descending). Length = `dim()`.
    pub fn eigvals(&self) -> ArrayView1<'_, f64> {
        self.eigvals.view()
    }
}

impl<B: Basis> BasisTransform for StableReparam<B> {
    type Inner = B;
    fn inner(&self) -> &B {
        &self.inner
    }
    fn matrix(&self) -> ArrayView2<'_, f64> {
        self.v.view()
    }
}

impl<B: Basis> Basis for StableReparam<B> {
    fn dim(&self) -> usize {
        self.v.ncols()
    }
    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }
    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.inner.evaluate(x).dot(&self.v)
    }
    fn d1(&self, x: ArrayView2<f64>, axis: usize) -> Array2<f64> {
        self.inner.d1(x, axis).dot(&self.v)
    }
    fn penalties(&self) -> Vec<Array2<f64>> {
        // Rotated penalty `V' S V` — diagonal up to FP. We build it
        // exactly (V' · S_inner · V) so any FP drift mirrors what the
        // downstream consumers will see.
        self.inner
            .penalties()
            .into_iter()
            .map(|s| self.v.t().dot(&s).dot(&self.v))
            .collect()
    }
}

/// Eigendecomposition of a symmetric penalty with the eigenpairs in
/// descending order, so the range space sits first (v0.x
/// `setup_initial_repara` / mgcv `eigen()` convention). The rotation
/// [`StableReparam`] applies, available without a `Basis` to wrap.
pub(crate) fn descending_eigh(penalty: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let k = penalty.nrows();
    let (eigvals_asc, v_asc) = penalty
        .to_owned()
        .eigh(UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(format!("eigh failed in StableReparam: {e}")))?;
    let mut eigvals = Array1::<f64>::zeros(k);
    let mut v = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        eigvals[j] = eigvals_asc[k - 1 - j];
        for i in 0..k {
            v[[i, j]] = v_asc[[i, k - 1 - j]];
        }
    }
    Ok((eigvals, v))
}

// =============================================================================
// Householder helper for SumToZero.
// =============================================================================

/// Householder reflector on `t`: returns the `(k, k-1)` matrix whose
/// columns are orthonormal and orthogonal to `t`. Equivalent to the last
/// `k-1` columns of `Q` from `QR(t)`.
/// Public wrapper around [`nullspace_householder`] for callers (e.g. the
/// n-margin tensor builder) that have a raw column-sum vector but no
/// `Basis` instance to feed [`SumToZero::from_fit_design`].
pub(crate) fn nullspace_householder_pub(t: ArrayView1<f64>) -> Array2<f64> {
    nullspace_householder(t)
}

fn nullspace_householder(t: ArrayView1<f64>) -> Array2<f64> {
    let k = t.len();
    debug_assert!(k >= 2);

    let norm = t.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm < f64::EPSILON {
        // Degenerate: t is zero. Return identity-like (k, k-1) basis —
        // shouldn't happen in practice (basis always sums to >0 over real data).
        let mut c = Array2::<f64>::zeros((k, k - 1));
        for j in 0..k - 1 {
            c[[j, j]] = 1.0;
        }
        return c;
    }

    // v = t + sign(t[0]) · ‖t‖ · e_1
    let sign = if t[0] >= 0.0 { 1.0 } else { -1.0 };
    let mut v = Array1::<f64>::zeros(k);
    v[0] = t[0] + sign * norm;
    for i in 1..k {
        v[i] = t[i];
    }
    let v_norm_sq = v.iter().map(|&x| x * x).sum::<f64>();

    // Q = I - 2 v v' / (v'v). We want columns 1..k of Q (skip the first).
    // Direct formula: Q[i,j] = δ_ij - 2 v[i] v[j] / v_norm_sq.
    let mut c = Array2::<f64>::zeros((k, k - 1));
    let scale = 2.0 / v_norm_sq;
    for j in 0..k - 1 {
        // Column index in Q is j+1 (we drop column 0).
        let q_col = j + 1;
        for i in 0..k {
            let delta = if i == q_col { 1.0 } else { 0.0 };
            c[[i, j]] = delta - scale * v[i] * v[q_col];
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::CrSpline;
    use ndarray::array;

    fn col(x: &Array1<f64>) -> Array2<f64> {
        x.view().insert_axis(ndarray::Axis(1)).to_owned()
    }

    #[test]
    fn constraint_zeroes_column_sum_of_centred_design() {
        let knots = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let cr = CrSpline::new(knots).unwrap();
        let x = Array1::linspace(0.0, 1.0, 50);
        let x2 = col(&x);
        let x_design = cr.evaluate(x2.view());
        let stz = SumToZero::from_fit_design(cr, x_design.view());

        let centred = stz.evaluate(x2.view());
        let colsums = centred.sum_axis(Axis(0));
        for (j, &cs) in colsums.iter().enumerate() {
            assert!(cs.abs() < 1e-9, "centred col {j} sums to {cs}, expected 0");
        }
        assert_eq!(centred.ncols(), 4); // k-1 = 5-1
    }

    #[test]
    fn constraint_columns_are_orthonormal() {
        let knots = array![0.0, 0.3, 0.7, 1.0];
        let cr = CrSpline::new(knots).unwrap();
        let x = Array1::linspace(0.0, 1.0, 30);
        let x2 = col(&x);
        let x_design = cr.evaluate(x2.view());
        let stz = SumToZero::from_fit_design(cr, x_design.view());
        let c = stz.matrix();
        let ctc = c.t().dot(&c);
        for i in 0..ctc.nrows() {
            for j in 0..ctc.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((ctc[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }

    // -----------------------------------------------------------------
    // StableReparam tests
    // -----------------------------------------------------------------

    /// Build the centred-block penalty and the SumToZero<CrSpline> basis
    /// from a fitted CR spline. Returns the basis + penalty needed by
    /// `StableReparam::from_inner_penalty`.
    fn build_centred_cr(knots: Array1<f64>, x: &Array1<f64>) -> (SumToZero<CrSpline>, Array2<f64>) {
        let cr = CrSpline::new(knots).unwrap();
        let x2 = col(x);
        let raw_design = cr.evaluate(x2.view());
        let stz = SumToZero::from_fit_design(cr, raw_design.view());
        let centred_penalty = stz.penalties().pop().unwrap();
        (stz, centred_penalty)
    }

    #[test]
    fn reparam_v_is_orthogonal() {
        let knots = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let x = Array1::linspace(0.0, 1.0, 80);
        let (stz, s_centred) = build_centred_cr(knots, &x);
        let reparam = StableReparam::from_inner_penalty(stz, s_centred.view()).unwrap();
        let v = reparam.matrix();
        let vtv = v.t().dot(&v);
        let k = vtv.nrows();
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vtv[[i, j]] - expected).abs() < 1e-10,
                    "V'V off at ({i}, {j}) = {} expected {expected}",
                    vtv[[i, j]]
                );
            }
        }
    }

    #[test]
    fn reparam_diagonalises_penalty() {
        let knots = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let x = Array1::linspace(0.0, 1.0, 100);
        let (stz, s_centred) = build_centred_cr(knots, &x);
        let reparam = StableReparam::from_inner_penalty(stz, s_centred.view()).unwrap();
        // After rotation, V' S V should be diag(eigvals) up to FP.
        let rotated_penalty = reparam.penalties().pop().unwrap();
        let k = rotated_penalty.nrows();
        let scale = reparam
            .eigvals()
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    let diff = (rotated_penalty[[i, j]] - reparam.eigvals()[i]).abs();
                    assert!(
                        diff < 1e-10 * scale,
                        "diag mismatch at ({i}): rotated={} eig={} diff={diff:.3e}",
                        rotated_penalty[[i, j]],
                        reparam.eigvals()[i]
                    );
                } else {
                    assert!(
                        rotated_penalty[[i, j]].abs() < 1e-10 * scale,
                        "off-diag ({i}, {j}) = {} expected ~0",
                        rotated_penalty[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn reparam_eigvals_are_descending() {
        let knots = array![0.0, 0.2, 0.5, 0.8, 1.0];
        let x = Array1::linspace(0.0, 1.0, 60);
        let (stz, s_centred) = build_centred_cr(knots, &x);
        let reparam = StableReparam::from_inner_penalty(stz, s_centred.view()).unwrap();
        let eigvals = reparam.eigvals();
        for j in 1..eigvals.len() {
            assert!(
                eigvals[j - 1] >= eigvals[j] - 1e-12,
                "eigvals not descending at j={j}: {} < {}",
                eigvals[j - 1],
                eigvals[j]
            );
        }
    }

    #[test]
    fn reparam_evaluate_composes_with_inner() {
        let knots = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let x = Array1::linspace(0.0, 1.0, 100);
        let (stz, s_centred) = build_centred_cr(knots.clone(), &x);
        // Re-build a parallel SumToZero so we can compare against it.
        let (stz_ref, _) = build_centred_cr(knots, &x);
        let reparam = StableReparam::from_inner_penalty(stz, s_centred.view()).unwrap();

        // Evaluate on a fresh grid.
        let x_new = Array1::linspace(0.1, 0.9, 25);
        let x_new_2 = col(&x_new);
        let centred_eval = stz_ref.evaluate(x_new_2.view());
        let rotated_eval = reparam.evaluate(x_new_2.view());
        let v = reparam.matrix();
        let expected = centred_eval.dot(&v);
        let max_diff = rotated_eval
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "rotated eval != centred·V (max_diff {max_diff:.3e})"
        );
    }

    #[test]
    fn reparam_dim_unchanged() {
        let knots = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let x = Array1::linspace(0.0, 1.0, 50);
        let (stz, s_centred) = build_centred_cr(knots, &x);
        let inner_dim = stz.dim();
        let reparam = StableReparam::from_inner_penalty(stz, s_centred.view()).unwrap();
        assert_eq!(reparam.dim(), inner_dim, "StableReparam preserves dim");
    }
}
