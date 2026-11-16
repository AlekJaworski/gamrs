//! Shape-constrained (monotone) smooths: mgcv's `s(x, bs = "sc", xt = "m+")`
//! and `"m-"`, as fitted by `scasm()`.
//!
//! The basis is mgcv's `bs` B-spline ([`BsBasis`], `m = c(3, 2)`) centred
//! like a CR term: the fit-time sum-to-zero row is absorbed into the basis
//! through an orthonormal Householder map `Z` (`β_raw = Z β`). The monotone
//! condition stays on the raw B-spline coefficients, where a B-spline is
//! increasing wherever its coefficients are: `±(β_raw,j+1 − β_raw,j) ≥ 0`.
//! On the absorbed coefficients those rows become `A Z β ≥ 0`. `Z` maps the
//! absorbed space one-to-one onto the raw coefficients that satisfy the
//! centring row, so the constrained problem — and its solution — is the
//! one scasm solves with the centring kept as an equality row.
//!
//! The rows travel to the fit through [`ShapeConstraints`] on the
//! prepared design; the Gaussian driver enforces them.

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::basis::BsBasis;
use crate::error::{GamrsError, Result};
use crate::traits::{Basis, BasisTransform};
use crate::transform::SumToZero;

use super::{matrix_inf_norm, prepend_intercept, prepend_zero_column};

/// Monotone direction of a shape-constrained smooth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(rename_all = "snake_case"))]
pub enum Shape {
    Increasing,
    Decreasing,
}

impl Shape {
    pub(crate) fn sign(self) -> f64 {
        match self {
            Shape::Increasing => 1.0,
            Shape::Decreasing => -1.0,
        }
    }
}

/// Linear inequality rows on a prepared design's coefficients:
/// `rows · β ≥ 0`, with a strictly feasible `start`. Only shape-constrained
/// terms contribute rows; every other coefficient has zero columns here.
#[derive(Debug, Clone)]
pub struct ShapeConstraints {
    /// `(m, p)` constraint rows over the full coefficient vector
    /// (intercept first).
    pub rows: Array2<f64>,
    /// `(p,)` coefficients satisfying every row strictly.
    pub start: Array1<f64>,
    /// One entry per constrained term: its rows `rows.0..rows.1` act only
    /// on its coefficients `cols.0..cols.1`.
    pub blocks: Vec<ConstraintBlock>,
}

/// Where one term's rows and coefficients sit in [`ShapeConstraints`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintBlock {
    pub rows: (usize, usize),
    pub cols: (usize, usize),
}

/// One term's rows and feasible start in that term's own coefficients.
pub(crate) struct TermConstraints {
    pub rows: Array2<f64>,
    pub start: Array1<f64>,
}

/// The shape rows of one `k`-coefficient B-spline: `±(β_{j+1} − β_j) ≥ 0`
/// (`smooth.construct.sc.smooth.spec`, `xt = "m+"` / `"m-"`).
pub(crate) fn shape_rows(shape: Shape, k: usize) -> Array2<f64> {
    let s = shape.sign();
    let mut a = Array2::<f64>::zeros((k - 1, k));
    for j in 0..k - 1 {
        a[[j, j]] = -s;
        a[[j, j + 1]] = s;
    }
    a
}

/// mgcv's `ucon`: scale each row so its first non-zero entry has magnitude
/// 1 and drop duplicates, keeping first occurrences in order.
pub(crate) fn unique_rows(a: Array2<f64>) -> Array2<f64> {
    let p = a.ncols();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(a.nrows());
    for row in a.axis_iter(Axis(0)) {
        let mut r: Vec<f64> = row.to_vec();
        if let Some(lead) = r.iter().find(|v| **v != 0.0).map(|v| v.abs()) {
            r.iter_mut().for_each(|v| *v /= lead);
        }
        if !rows.contains(&r) {
            rows.push(r);
        }
    }
    let m = rows.len();
    Array2::from_shape_vec((m, p), rows.into_iter().flatten().collect())
        .expect("rows all have length p")
}

/// A strictly feasible start for a monotone 1-D smooth in raw B-spline
/// coefficients: successive coefficients differ by exactly `ε^{1/4}` in the
/// constrained direction, shifted so `centre · β = 0`. This is the vertex
/// mgcv's phase-1 LP (`feasible`) returns for these constraints.
fn monotone_start(shape: Shape, centre: &Array1<f64>) -> Array1<f64> {
    let step = shape.sign() * f64::EPSILON.powf(0.25);
    let k = centre.len();
    let ramp: Array1<f64> = (0..k).map(|j| step * j as f64).collect();
    let total: f64 = centre.sum();
    let level = if total != 0.0 {
        -centre.dot(&ramp) / total
    } else {
        0.0
    };
    ramp.mapv(|v| v + level)
}

/// Predict-time rebuilder for a 1-D shape-constrained smooth: the B-spline
/// basis and the centring map.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct ScPredictor {
    pub basis: BsBasis,
    pub shape: Shape,
    /// `(k, k − 1)` sum-to-zero map `Z`.
    pub centring: Array2<f64>,
}

impl ScPredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let centred = self.basis.evaluate(x_new).dot(&self.centring);
        Ok(prepend_intercept(centred.view()))
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        let centred = self.basis.d1(x_new, axis).dot(&self.centring);
        Ok(prepend_zero_column(centred.view()))
    }

    /// Centred design, penalty and constraint rows of `s(x, bs = "sc")` on
    /// column `col` of `x`. The penalty is rescaled on the raw basis as for
    /// a CR term (mgcv's `scale.penalty`), then rotated by `Z`.
    pub(crate) fn fit_centred(
        x: ArrayView2<f64>,
        col: usize,
        k: usize,
        shape: Shape,
    ) -> Result<CentredScFit> {
        let basis = BsBasis::from_data(x.column(col), k, (3, 2))?;
        let raw = basis.evaluate(x.slice(ndarray::s![.., col..col + 1]));
        let s_raw = rescale_penalty(basis.penalty(), raw.view());
        let start_raw = monotone_start(shape, &raw.sum_axis(Axis(0)));
        let centring = SumToZero::from_fit_design(basis.clone(), raw.view())
            .matrix()
            .to_owned();
        let centred = raw.dot(&centring);
        let s_smooth = centring.t().dot(&s_raw).dot(&centring);
        let constraints = TermConstraints {
            rows: unique_rows(shape_rows(shape, k)).dot(&centring),
            start: centring.t().dot(&start_raw),
        };
        Ok(CentredScFit {
            predictor: ScPredictor {
                basis,
                shape,
                centring,
            },
            centred,
            s_smooth,
            constraints,
        })
    }
}

pub(crate) struct CentredScFit {
    pub predictor: ScPredictor,
    pub centred: Array2<f64>,
    pub s_smooth: Array2<f64>,
    pub constraints: TermConstraints,
}

/// mgcv's `scale.penalty` as the CR path applies it: `S · ‖X‖∞² / ‖S‖∞`.
pub(crate) fn rescale_penalty(s: Array2<f64>, raw_design: ArrayView2<f64>) -> Array2<f64> {
    let x_inf = matrix_inf_norm(raw_design);
    let s_inf = matrix_inf_norm(s.view());
    if s_inf > 0.0 {
        s * ((x_inf * x_inf) / s_inf)
    } else {
        s
    }
}

/// One margin's shape rows `a` (acting on that margin's `k[j]`
/// coefficients) placed in a two-margin tensor's `k[0]·k[1]` coefficients,
/// margin 0 varying slowest: `a ⊗ I` for margin 0, `I ⊗ a` for margin 1.
pub(crate) fn tensor_margin_rows(a: &Array2<f64>, j: usize, k: [usize; 2]) -> Array2<f64> {
    let (k0, k1) = (k[0], k[1]);
    let m = a.nrows();
    let mut out;
    if j == 0 {
        out = Array2::<f64>::zeros((m * k1, k0 * k1));
        for r in 0..m {
            for c in 0..k0 {
                for t in 0..k1 {
                    out[[r * k1 + t, c * k1 + t]] = a[[r, c]];
                }
            }
        }
    } else {
        out = Array2::<f64>::zeros((k0 * m, k0 * k1));
        for s in 0..k0 {
            for r in 0..m {
                for c in 0..k1 {
                    out[[s * m + r, s * k1 + c]] = a[[r, c]];
                }
            }
        }
    }
    out
}

/// Error for a request the shape-constrained path does not cover.
pub(crate) fn unsupported(what: &str) -> GamrsError {
    GamrsError::InvalidParameter(format!(
        "{what} is not supported with shape-constrained (sc) terms"
    ))
}
