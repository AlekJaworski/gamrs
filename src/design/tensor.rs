//! Tensor-product term — fit-time wiring + predict-time rebuilder.
//!
//! Layout matches mgcv's `te(x_a, x_b)`
//! (`smooth.construct.tensor.smooth.spec` in `mgcv/R/smooth.r`):
//! marginal bases are UNCENTRED (`mc[i] = FALSE` for every margin of a
//! plain `te`), the row-wise Kronecker is taken, per-margin `Sm[i]` are
//! rescaled by their leading eigenvalue, the tensor-product penalties
//! `tensor.prod.penalties(Sm) = [S_a ⊗ I_b, I_a ⊗ S_b]` are built in
//! the uncentred frame, then a SINGLE sum-to-zero constraint is applied
//! to the WHOLE tensor product (via `absorb.cons` at the smoothCon
//! level). All three — design + two penalties — are rotated through
//! the constraint matrix `C`, dropping one column.
//!
//! Net dim after centring: `k_a · k_b − 1` smooth columns (+1 for the
//! intercept handled by the [`super::Additive`] driver).
//!
//! Lives in its own submodule (split from `design/additive.rs` once the
//! tensor wiring pushed `additive.rs` over the 700-LOC threshold).

use ndarray::{Array2, ArrayView2};

use crate::basis::{CrSpline, TensorProductBasis};
use crate::error::{GamrsError, Result};
use crate::traits::{Basis, BasisTransform};
use crate::transform::SumToZero;

use super::additive::MarginKind;
use super::by::{scale_rows, varies};
use super::{prepend_intercept, prepend_zero_column};

/// Predict-time tensor product rebuilder. Carries the marginal knot
/// grids + each margin's `XP` reparameterisation matrix (mgcv smooth.r
/// line 817) + the tensor-level centring matrix `C` so the design can be
/// rebuilt on new x.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorPredictor {
    pub col_a: usize,
    pub col_b: usize,
    pub bs_a: MarginKind,
    pub bs_b: MarginKind,
    /// Margin A knot grid (currently always CR — `bs_a = MarginKind::Cr`).
    pub knots_a: ndarray::Array1<f64>,
    /// Margin B knot grid.
    pub knots_b: ndarray::Array1<f64>,
    /// Margin A reparameterisation `XP_a = V_a Σ_a⁻¹ U_a'` (from SVD of
    /// the marginal design at uniformly-spaced eval points; mgcv
    /// smooth.r line 817). Shape `(k_a, k_a)`. Applied as
    /// `B_a_new = B_a_raw · XP_a` before the tensor product. For CR
    /// margins (which set `noterp = TRUE`) this is the identity matrix.
    pub xp_a: Array2<f64>,
    /// Margin B reparameterisation, same shape and role as `xp_a`.
    pub xp_b: Array2<f64>,
    /// Tensor-level sum-to-zero centring matrix, shape
    /// `(k_a*k_b, k_a*k_b - 1)`, or the identity for a term with a varying
    /// `by=` variable. Applied AFTER the row-wise Kronecker of the
    /// (reparameterised) marginal designs.
    pub centring: Array2<f64>,
}

impl TensorPredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check_cols(x_new)?;
        let raw = self.tensor_design(x_new, /*deriv_axis=*/ None);
        let centred = raw.dot(&self.centring);
        Ok(prepend_intercept(centred.view()))
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        self.check_cols(x_new)?;
        let raw_d1 = self.tensor_design(x_new, Some(axis));
        let centred_d1 = raw_d1.dot(&self.centring);
        Ok(prepend_zero_column(centred_d1.view()))
    }

    fn check_cols(&self, x_new: ArrayView2<f64>) -> Result<()> {
        if self.col_a >= x_new.ncols() || self.col_b >= x_new.ncols() {
            return Err(GamrsError::InvalidParameter(format!(
                "TensorPredictor: term reads columns ({}, {}) but x has only {} columns",
                self.col_a,
                self.col_b,
                x_new.ncols()
            )));
        }
        Ok(())
    }

    /// Build the row-wise Kronecker of the (reparameterised) marginals.
    /// If `deriv_axis` is `Some(axis)`, the corresponding margin's
    /// derivative is used (and the other margin's design is unchanged);
    /// if `axis` matches neither column, the result is a zero matrix.
    fn tensor_design(&self, x_new: ArrayView2<f64>, deriv_axis: Option<usize>) -> Array2<f64> {
        let (MarginKind::Cr, MarginKind::Cr) = (self.bs_a, self.bs_b);
        let cr_a = CrSpline::new(self.knots_a.clone()).expect("invalid stored knots_a");
        let cr_b = CrSpline::new(self.knots_b.clone()).expect("invalid stored knots_b");
        let x_a = x_new.slice(ndarray::s![.., self.col_a..self.col_a + 1]);
        let x_b = x_new.slice(ndarray::s![.., self.col_b..self.col_b + 1]);

        // Margin A: apply XP_a to raw design (or to raw d1 if differentiating wrt col_a).
        let raw_a = match deriv_axis {
            Some(ax) if ax == self.col_a => cr_a.d1(x_a, 0),
            _ => cr_a.evaluate(x_a),
        };
        let mut design_a = raw_a.dot(&self.xp_a);

        // Margin B: same logic, differentiating wrt col_b.
        let raw_b = match deriv_axis {
            Some(ax) if ax == self.col_b => cr_b.d1(x_b, 0),
            _ => cr_b.evaluate(x_b),
        };
        let design_b = raw_b.dot(&self.xp_b);

        // If deriv_axis is Some but matches neither column, the result
        // is identically zero.
        if let Some(ax) = deriv_axis {
            if ax != self.col_a && ax != self.col_b {
                design_a.fill(0.0);
            }
        }

        row_kron(design_a.view(), design_b.view())
    }
}

/// Fit-time helper: build a tensor-product term's centred design,
/// per-margin penalties (already in the centred coefficient frame),
/// and predict-time rebuilder. Mirrors mgcv's
/// `smooth.construct.tensor.smooth.spec` for a 2-margin plain `te()`.
pub(super) struct TensorTermFit {
    /// `(n, k_a*k_b - 1)` design in the centred frame.
    pub(super) design: Array2<f64>,
    /// Per-margin penalties in the centred frame. Length 2:
    /// `[C' (S_a ⊗ I_{k_b}) C, C' (I_{k_a} ⊗ S_b) C]`.
    pub(super) s_list_term: Vec<Array2<f64>>,
    pub(super) predictor: TensorPredictor,
}

impl TensorTermFit {
    pub(super) fn build(
        x: ArrayView2<f64>,
        col_a: usize,
        col_b: usize,
        k_a: usize,
        k_b: usize,
        bs_a: MarginKind,
        bs_b: MarginKind,
        by: Option<usize>,
    ) -> Result<Self> {
        if col_a == col_b {
            return Err(GamrsError::InvalidParameter(format!(
                "Tensor term must have distinct margin columns (got col_a == col_b == {col_a})"
            )));
        }
        let (MarginKind::Cr, MarginKind::Cr) = (bs_a, bs_b);
        // Uncentred marginal bases (matches mgcv's `mc[i] = FALSE` for
        // every margin of a plain `te`).
        let x_a_col = x.column(col_a);
        let x_b_col = x.column(col_b);
        let cr_a = CrSpline::with_quantile_knots(x_a_col, k_a)?;
        let cr_b = CrSpline::with_quantile_knots(x_b_col, k_b)?;
        let knots_a = cr_a.knots().to_owned();
        let knots_b = cr_b.knots().to_owned();

        // CR margins set `noterp = TRUE` (smooth.r:1512), which causes
        // smooth.construct.tensor.smooth.spec to SKIP the SVD repara
        // (smooth.r:799 — `if (is.null(object$margin[[i]]$noterp))`).
        // For CR + CR `te` we therefore use identity reparams; if we
        // ever add B-spline marginals here we'd switch back to the SVD
        // path conditional on the margin kind.
        let xp_a = Array2::<f64>::eye(k_a);
        let xp_b = Array2::<f64>::eye(k_b);

        // Marginal penalties in the reparameterised frame (identity
        // here since CR margins are noterp). mgcv smooth.r:825 rescales
        // each marginal penalty by its leading eigenvalue.
        let s_a_raw = cr_a.penalties().pop().unwrap();
        let s_b_raw = cr_b.penalties().pop().unwrap();
        let s_a = rescale_by_leading_eig(s_a_raw)?;
        let s_b = rescale_by_leading_eig(s_b_raw)?;

        // Evaluate the marginal designs on x (no XP rotation for CR
        // since xp = I). Row-wise Kronecker → tensor product design.
        let n = x.nrows();
        let x_a_view = x.slice(ndarray::s![.., col_a..col_a + 1]);
        let x_b_view = x.slice(ndarray::s![.., col_b..col_b + 1]);
        let design_a = cr_a.evaluate(x_a_view);
        let design_b = cr_b.evaluate(x_b_view);
        debug_assert_eq!(design_a.shape(), &[n, k_a]);
        debug_assert_eq!(design_b.shape(), &[n, k_b]);

        let raw_design = row_kron(design_a.view(), design_b.view());

        // Tensor product penalties on the reparameterised + rescaled
        // marginal frame: S_te_a = S_a ⊗ I_{k_b}, S_te_b = I_{k_a} ⊗ S_b.
        let pk = k_a * k_b;
        let s_te_a_raw = kron_with_identity_right(s_a.view(), k_b);
        let s_te_b_raw = kron_with_identity_left(k_a, s_b.view());
        debug_assert_eq!(s_te_a_raw.shape(), &[pk, pk]);
        debug_assert_eq!(s_te_b_raw.shape(), &[pk, pk]);

        // Single sum-to-zero constraint on the WHOLE tensor product
        // (mgcv's `smoothCon` with `absorb.cons = TRUE`). The
        // Householder construction in `SumToZero::from_fit_design`
        // builds C from `colSums(X_fit)`, giving an orthonormal C of
        // shape `(k_a*k_b, k_a*k_b - 1)`.
        // A varying numeric `by=` drops the constraint, as for a CR term.
        let w = by.map(|b| x.column(b));
        let centring = if w.is_some_and(varies) {
            Array2::eye(pk)
        } else {
            let te = TensorProductBasis::new(cr_a, cr_b, col_a, col_b);
            SumToZero::from_fit_design(te, raw_design.view())
                .matrix()
                .to_owned()
        };
        let mut design = raw_design.dot(&centring);
        scale_rows(&mut design, w);

        // Rotate penalties through C: S_centred = C' · S_raw · C.
        let s_te_a = centring.t().dot(&s_te_a_raw).dot(&centring);
        let s_te_b = centring.t().dot(&s_te_b_raw).dot(&centring);

        let predictor = TensorPredictor {
            col_a,
            col_b,
            bs_a,
            bs_b,
            knots_a,
            knots_b,
            xp_a,
            xp_b,
            centring,
        };

        Ok(Self {
            design,
            s_list_term: vec![s_te_a, s_te_b],
            predictor,
        })
    }
}

// =============================================================================
// N-margin tensor product (te) and tensor interaction (ti).
// =============================================================================

/// Predict-time rebuilder for an n-margin tensor product (`te`) or tensor
/// interaction (`ti`). Holds each margin's knot grid + optional per-margin
/// centring matrix (used by `ti` to strip each margin's main effect) and
/// the single tensor-level sum-to-zero centring matrix.
///
/// For `te` the per-margin centrings are absent (uncentred marginals,
/// matching mgcv `mc[i] = FALSE`); for `ti` each margin carries an
/// `(k_i, k_i - 1)` sum-to-zero matrix so the row-wise Kronecker contains
/// only the pure interaction.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorMultiPredictor {
    /// Column index in `x` each margin reads. Length `D >= 2`.
    pub cols: Vec<usize>,
    /// Per-margin knot grids (currently all CR). Length `D`.
    pub knots: Vec<ndarray::Array1<f64>>,
    /// Optional per-margin sum-to-zero centring matrix. `None` for `te`
    /// (uncentred marginals); `Some((k_i, k_i - 1))` for `ti`. Length `D`.
    pub margin_centrings: Vec<Option<Array2<f64>>>,
    /// Tensor-level sum-to-zero centring matrix, shape
    /// `(prod_k, prod_k - 1)`. Applied AFTER the row-wise Kronecker.
    pub centring: Array2<f64>,
}

impl TensorMultiPredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.check_cols(x_new)?;
        let raw = self.tensor_design(x_new, /*deriv_axis=*/ None);
        let centred = raw.dot(&self.centring);
        Ok(prepend_intercept(centred.view()))
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        self.check_cols(x_new)?;
        let raw_d1 = self.tensor_design(x_new, Some(axis));
        let centred_d1 = raw_d1.dot(&self.centring);
        Ok(prepend_zero_column(centred_d1.view()))
    }

    fn check_cols(&self, x_new: ArrayView2<f64>) -> Result<()> {
        for &c in &self.cols {
            if c >= x_new.ncols() {
                return Err(GamrsError::InvalidParameter(format!(
                    "TensorMultiPredictor: term reads column {c} but x has only {} columns",
                    x_new.ncols()
                )));
            }
        }
        Ok(())
    }

    /// Row-wise Kronecker of the (optionally centred) marginal designs. If
    /// `deriv_axis` is `Some(axis)` the margin reading that column uses its
    /// derivative; if `axis` matches no margin column the result is zero.
    fn tensor_design(&self, x_new: ArrayView2<f64>, deriv_axis: Option<usize>) -> Array2<f64> {
        let mut acc: Option<Array2<f64>> = None;
        let mut any_deriv_margin = false;
        for (m, &col) in self.cols.iter().enumerate() {
            let cr = CrSpline::new(self.knots[m].clone()).expect("invalid stored knots");
            let x_col = x_new.slice(ndarray::s![.., col..col + 1]);
            let raw = match deriv_axis {
                Some(ax) if ax == col => {
                    any_deriv_margin = true;
                    cr.d1(x_col, 0)
                }
                _ => cr.evaluate(x_col),
            };
            let margin_design = match &self.margin_centrings[m] {
                Some(cm) => raw.dot(cm),
                None => raw,
            };
            acc = Some(match acc {
                None => margin_design,
                Some(prev) => row_kron(prev.view(), margin_design.view()),
            });
        }
        let mut out = acc.expect("tensor must have >= 1 margin");
        // If differentiating wrt an axis that no margin reads, the product
        // rule gives identically zero.
        if let Some(_ax) = deriv_axis {
            if !any_deriv_margin {
                out.fill(0.0);
            }
        }
        out
    }
}

/// Fit-time helper for n-margin tensor products. Mirrors
/// [`TensorTermFit`] but generalises to `D >= 2` margins and supports both
/// `te` (uncentred marginals) and `ti` (per-margin sum-to-zero so main
/// effects are excluded).
pub(super) struct TensorMultiTermFit {
    /// `(n, prod_k - 1)` design in the centred frame.
    pub(super) design: Array2<f64>,
    /// Per-margin penalties in the centred frame. Length `D` (one
    /// smoothing parameter per margin).
    pub(super) s_list_term: Vec<Array2<f64>>,
    pub(super) predictor: TensorMultiPredictor,
}

impl TensorMultiTermFit {
    /// Build an n-margin `te(...)` term (uncentred marginals).
    pub(super) fn build_te(
        x: ArrayView2<f64>,
        cols: &[usize],
        k: &[usize],
        bs: &[MarginKind],
    ) -> Result<Self> {
        Self::build_impl(x, cols, k, bs, /*interaction=*/ false)
    }

    /// Build an n-margin `ti(...)` term — pure interaction with each
    /// margin's main effect removed via a per-margin sum-to-zero
    /// constraint applied BEFORE the Kronecker product (mgcv `ti`'s
    /// `mc[i] = TRUE` for every margin).
    pub(super) fn build_ti(
        x: ArrayView2<f64>,
        cols: &[usize],
        k: &[usize],
        bs: &[MarginKind],
    ) -> Result<Self> {
        Self::build_impl(x, cols, k, bs, /*interaction=*/ true)
    }

    fn build_impl(
        x: ArrayView2<f64>,
        cols: &[usize],
        k: &[usize],
        bs: &[MarginKind],
        interaction: bool,
    ) -> Result<Self> {
        let d = cols.len();
        if d < 2 {
            return Err(GamrsError::InvalidParameter(format!(
                "n-margin tensor term must have >= 2 margins (got {d})"
            )));
        }
        if k.len() != d || bs.len() != d {
            return Err(GamrsError::InvalidParameter(format!(
                "n-margin tensor term: cols ({d}), k ({}), bs ({}) lengths must match",
                k.len(),
                bs.len()
            )));
        }
        // Distinct columns.
        for i in 0..d {
            for j in (i + 1)..d {
                if cols[i] == cols[j] {
                    return Err(GamrsError::InvalidParameter(format!(
                        "n-margin tensor term must have distinct margin columns \
                         (cols[{i}] == cols[{j}] == {})",
                        cols[i]
                    )));
                }
            }
        }
        for &b in bs {
            let MarginKind::Cr = b;
        }

        // Per-margin: build CR basis, evaluate raw design, optionally apply
        // a per-margin sum-to-zero constraint (ti), and form the rescaled
        // marginal penalty in the (possibly centred) marginal frame.
        let mut knots: Vec<ndarray::Array1<f64>> = Vec::with_capacity(d);
        let mut margin_centrings: Vec<Option<Array2<f64>>> = Vec::with_capacity(d);
        let mut margin_designs: Vec<Array2<f64>> = Vec::with_capacity(d);
        let mut margin_penalties: Vec<Array2<f64>> = Vec::with_capacity(d);
        let mut margin_dims: Vec<usize> = Vec::with_capacity(d);

        for m in 0..d {
            let x_col = x.column(cols[m]);
            let cr = CrSpline::with_quantile_knots(x_col, k[m])?;
            knots.push(cr.knots().to_owned());
            let x_view = x.slice(ndarray::s![.., cols[m]..cols[m] + 1]);
            let raw_design = cr.evaluate(x_view);
            let s_raw = cr.penalties().pop().unwrap();
            let s_rescaled = rescale_by_leading_eig(s_raw)?;

            if interaction {
                // ti: sum-to-zero on this margin so its main effect drops.
                let cr_for_stz = CrSpline::new(cr.knots().to_owned())?;
                let stz = SumToZero::from_fit_design(cr_for_stz, raw_design.view());
                let cm = stz.matrix().to_owned();
                let centred_design = raw_design.dot(&cm);
                let centred_pen = cm.t().dot(&s_rescaled).dot(&cm);
                margin_dims.push(cm.ncols());
                margin_designs.push(centred_design);
                margin_penalties.push(centred_pen);
                margin_centrings.push(Some(cm));
            } else {
                // te: uncentred marginal (mgcv mc[i] = FALSE).
                margin_dims.push(k[m]);
                margin_designs.push(raw_design);
                margin_penalties.push(s_rescaled);
                margin_centrings.push(None);
            }
        }

        // Row-wise Kronecker over all margins → raw tensor design.
        let mut raw_design: Option<Array2<f64>> = None;
        for md in &margin_designs {
            raw_design = Some(match raw_design {
                None => md.clone(),
                Some(prev) => row_kron(prev.view(), md.view()),
            });
        }
        let raw_design = raw_design.expect(">= 2 margins guaranteed above");
        let prod_k: usize = margin_dims.iter().product();
        debug_assert_eq!(raw_design.ncols(), prod_k);

        // Per-margin tensor penalties: S_m lifted to the full product space
        // as I ⊗ ... ⊗ S_m ⊗ ... ⊗ I (S_m in slot m).
        let mut s_te_raw: Vec<Array2<f64>> = Vec::with_capacity(d);
        for m in 0..d {
            let left: usize = margin_dims[..m].iter().product();
            let right: usize = margin_dims[(m + 1)..].iter().product();
            let s_lifted = kron_identity_sandwich(left, margin_penalties[m].view(), right);
            debug_assert_eq!(s_lifted.shape(), &[prod_k, prod_k]);
            s_te_raw.push(s_lifted);
        }

        // Single sum-to-zero constraint on the WHOLE tensor product.
        let t = raw_design.sum_axis(ndarray::Axis(0));
        let centring = crate::transform::nullspace_householder_pub(t.view());
        let design = raw_design.dot(&centring);

        // Rotate penalties through the tensor-level constraint.
        let s_list_term: Vec<Array2<f64>> = s_te_raw
            .iter()
            .map(|s| centring.t().dot(s).dot(&centring))
            .collect();

        let predictor = TensorMultiPredictor {
            cols: cols.to_vec(),
            knots,
            margin_centrings,
            centring,
        };

        Ok(Self {
            design,
            s_list_term,
            predictor,
        })
    }
}

// =============================================================================
// Tensor-specific math helpers.
// =============================================================================

/// Row-wise Kronecker product: row `i` of the result is
/// `kron(A[i, :], B[i, :])` — matches mgcv `tensor.prod.model.matrix`.
/// Output shape: `(n, k_a * k_b)` with column index `j_a * k_b + j_b`.
fn row_kron(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let n = a.nrows();
    debug_assert_eq!(b.nrows(), n);
    let k_a = a.ncols();
    let k_b = b.ncols();
    let mut out = Array2::<f64>::zeros((n, k_a * k_b));
    for i in 0..n {
        for j_a in 0..k_a {
            let a_val = a[[i, j_a]];
            let base = j_a * k_b;
            for j_b in 0..k_b {
                out[[i, base + j_b]] = a_val * b[[i, j_b]];
            }
        }
    }
    out
}

/// Rescale `S` by its largest (positive) eigenvalue. Mirrors mgcv
/// smooth.r:825 — needed for tensor product penalty conditioning.
fn rescale_by_leading_eig(s: Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::Eigh;
    let (eigs, _) = s
        .clone()
        .eigh(ndarray_linalg::UPLO::Lower)
        .map_err(|e| GamrsError::Linalg(format!("eigh failed in tensor penalty rescale: {e}")))?;
    let max_eig = eigs.iter().cloned().fold(0.0_f64, f64::max);
    if max_eig <= 0.0 {
        return Ok(s);
    }
    Ok(s / max_eig)
}

/// Kronecker product `S ⊗ I_k`. Produces an `(s_rows*k, s_cols*k)` block
/// matrix where the `(i, j)`-th block is `S[i, j] · I_k`.
fn kron_with_identity_right(s: ndarray::ArrayView2<f64>, k: usize) -> Array2<f64> {
    let m = s.nrows();
    let n = s.ncols();
    let mut out = Array2::<f64>::zeros((m * k, n * k));
    for i in 0..m {
        for j in 0..n {
            let v = s[[i, j]];
            if v == 0.0 {
                continue;
            }
            for r in 0..k {
                out[[i * k + r, j * k + r]] = v;
            }
        }
    }
    out
}

/// Kronecker sandwich `I_left ⊗ S ⊗ I_right`. Produces a
/// `(left*m*right, left*n*right)` matrix used to lift a single margin's
/// `(m, n)` penalty into the full n-margin tensor-product coefficient
/// space. Column index in the product space is `((a * m) + i) * right + r`
/// for left-block `a`, S-row/col `i`, right-block `r`.
fn kron_identity_sandwich(left: usize, s: ndarray::ArrayView2<f64>, right: usize) -> Array2<f64> {
    let m = s.nrows();
    let n = s.ncols();
    let rows = left * m * right;
    let cols = left * n * right;
    let mut out = Array2::<f64>::zeros((rows, cols));
    for a in 0..left {
        for i in 0..m {
            for j in 0..n {
                let v = s[[i, j]];
                if v == 0.0 {
                    continue;
                }
                let row_base = (a * m + i) * right;
                let col_base = (a * n + j) * right;
                for r in 0..right {
                    out[[row_base + r, col_base + r]] = v;
                }
            }
        }
    }
    out
}

/// Kronecker product `I_k ⊗ S`. Produces an `(k*s_rows, k*s_cols)` block
/// matrix where the `(i, i)`-th block is `S` and off-diagonal blocks zero.
fn kron_with_identity_left(k: usize, s: ndarray::ArrayView2<f64>) -> Array2<f64> {
    let m = s.nrows();
    let n = s.ncols();
    let mut out = Array2::<f64>::zeros((k * m, k * n));
    for b in 0..k {
        let row_base = b * m;
        let col_base = b * n;
        for i in 0..m {
            for j in 0..n {
                out[[row_base + i, col_base + j]] = s[[i, j]];
            }
        }
    }
    out
}
