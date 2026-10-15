//! CR-spline DesignStrategy + Predictor (single-smooth, unrotated and
//! StableReparam variants).
//!
//! The two `DesignStrategy` impls here (`Cr`, `CrStable`) drive
//! `gamrs::fit(...)` and `fit_with_design(..., CrStable { k }, …)`. Both
//! produce a `PreparedDesign` with `s_list.len() == 1`; the multi-smooth
//! path lives in [`crate::design::Additive`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::basis::CrSpline;
use crate::error::{GamrsError, Result};
use crate::traits::{Basis, BasisTransform};
use crate::transform::{descending_eigh, SumToZero};

use super::by::scale_rows;
use super::{
    matrix_inf_norm, prepend_intercept, prepend_zero_column, rank_and_log_pseudo_det,
    DesignStrategy, Predictor, PreparedDesign,
};

/// Absorb the identifiability constraint into a CR basis: the fit-time
/// centering row by default, or mgcv's point constraint when `pc` is set.
fn constrain(cr: CrSpline, raw_design: ArrayView2<f64>, pc: Option<f64>) -> SumToZero<CrSpline> {
    match pc {
        Some(v) => SumToZero::from_point_constraint(cr, Array2::from_elem((1, 1), v).view()),
        None => SumToZero::from_fit_design(cr, raw_design),
    }
}

/// The coefficient map that absorbs a CR term's identifiability
/// constraint, or the identity when the term needs none. mgcv's
/// `smoothCon` drops the constraint for a numeric `by=` variable that
/// varies, since `w·f(x)` is then no longer confounded with the
/// intercept; a constant `w` keeps it.
fn constraint_map(
    cr: CrSpline,
    raw_design: ArrayView2<f64>,
    pc: Option<f64>,
    by: Option<ArrayView1<f64>>,
) -> Result<Array2<f64>> {
    if pc.is_some() && by.is_some() {
        return Err(GamrsError::InvalidParameter(
            "a CR term takes pc or by, not both: mgcv drops the point constraint \
             whenever the by-variable varies"
                .into(),
        ));
    }
    if by.is_some_and(varies) {
        return Ok(Array2::eye(cr.dim()));
    }
    Ok(constrain(cr, raw_design, pc).matrix().to_owned())
}

/// mgcv's `sd(by) > mean(by) * .Machine$double.eps * 1000`, on `|mean|`:
/// with the signed mean a constant negative `w` would count as varying and
/// leave a constant inside the smooth, aliased with the intercept.
fn varies(w: ArrayView1<f64>) -> bool {
    let n = w.len() as f64;
    let mean = w.sum() / n;
    let ss: f64 = w.iter().map(|v| (v - mean).powi(2)).sum();
    (ss / (n - 1.0)).sqrt() > mean.abs() * f64::EPSILON * 1000.0
}

// -----------------------------------------------------------------------------
// CR predictor — knots + sum-to-zero centring matrix.
// -----------------------------------------------------------------------------

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct CrPredictor {
    pub knots: Array1<f64>,
    pub centring: Array2<f64>,
}

impl CrPredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let cr = CrSpline::new(self.knots.clone())?;
        let raw = cr.evaluate(x_new);
        let centred = raw.dot(&self.centring);
        Ok(prepend_intercept(centred.view()))
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        let cr = CrSpline::new(self.knots.clone())?;
        let raw_d1 = cr.d1(x_new, axis);
        let centred_d1 = raw_d1.dot(&self.centring);
        Ok(prepend_zero_column(centred_d1.view()))
    }

    /// Build the centred design (no intercept column) and centring matrix
    /// for a CR smooth fit on `x_col`. Shared with `Additive`, which
    /// composes the centred designs from multiple terms.
    ///
    /// `pc` is mgcv's `s(x, pc=)`: with `Some(v)` the smooth is pinned to
    /// zero at `x = v` instead of centred over the fit rows. `by` is mgcv's
    /// numeric `s(x, by=w)`: each design row is scaled by `x[i, by]`.
    pub(crate) fn fit_centred(
        x: ArrayView2<f64>,
        x_col_idx: usize,
        k: usize,
        pc: Option<f64>,
        by: Option<usize>,
    ) -> Result<CenteredCrFit> {
        let x_col = x.column(x_col_idx);
        let cr = CrSpline::with_quantile_knots(x_col, k)?;
        let knots = cr.knots().to_owned();
        let x_view = x.slice(ndarray::s![.., x_col_idx..x_col_idx + 1]);
        let raw_design = cr.evaluate(x_view);
        let s_raw = cr.penalties().pop().unwrap();

        let x_inf = matrix_inf_norm(raw_design.view());
        let s_inf = matrix_inf_norm(s_raw.view());
        let rescale = if s_inf > 0.0 {
            (x_inf * x_inf) / s_inf
        } else {
            1.0
        };
        let s_raw_rescaled = &s_raw * rescale;

        let w = by.map(|b| x.column(b));
        let centring = constraint_map(cr, raw_design.view(), pc, w)?;
        let mut centred = raw_design.dot(&centring);
        // After the penalty rescale: mgcv sizes it on the unscaled basis.
        scale_rows(&mut centred, w);
        let s_smooth = centring.t().dot(&s_raw_rescaled).dot(&centring);

        Ok(CenteredCrFit {
            knots,
            centring,
            centred,
            s_smooth,
        })
    }
}

/// Internal helper carrying the pieces of a centred CR fit. Used by the
/// single-term `Cr::prepare` impl and by `Additive::prepare`.
pub(crate) struct CenteredCrFit {
    pub knots: Array1<f64>,
    pub centring: Array2<f64>,
    pub centred: Array2<f64>,
    pub s_smooth: Array2<f64>,
}

/// Cubic-regression spline + sum-to-zero constraint + intercept.
/// Default basis for [`gamrs::fit`](crate::fit).
pub struct Cr {
    pub k: usize,
}

impl DesignStrategy for Cr {
    fn prepare(&self, x: ArrayView2<f64>) -> Result<PreparedDesign> {
        let fit = CrPredictor::fit_centred(x, 0, self.k, None, None)?;
        let x_design = prepend_intercept(fit.centred.view());
        let p = x_design.ncols();
        let k_smooth = fit.centred.ncols();

        let mut s_block = Array2::<f64>::zeros((p, p));
        for i in 0..k_smooth {
            for j in 0..k_smooth {
                s_block[[1 + i, 1 + j]] = fit.s_smooth[[i, j]];
            }
        }

        let (rank_s, log_pseudo_det_s) = rank_and_log_pseudo_det(s_block.view())?;
        let mp = p - rank_s;

        Ok(PreparedDesign {
            x_design,
            s_list: vec![s_block],
            rank_s_list: vec![rank_s],
            log_pseudo_det_s_list: vec![log_pseudo_det_s],
            mp,
            predictor: Predictor::Cr(CrPredictor {
                knots: fit.knots,
                centring: fit.centring,
            }),
        })
    }
}

// -----------------------------------------------------------------------------
// CR + StableReparam predictor — knots, centring, AND rotation V.
// -----------------------------------------------------------------------------

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct CrStablePredictor {
    pub knots: Array1<f64>,
    pub centring: Array2<f64>,
    /// Eigen-rotation `V` from `StableReparam`. Columns are eigenvectors
    /// of the centred-block penalty in descending eigenvalue order.
    pub reparam_v: Array2<f64>,
}

impl CrStablePredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let cr = CrSpline::new(self.knots.clone())?;
        let raw = cr.evaluate(x_new);
        let rotated = raw.dot(&self.centring).dot(&self.reparam_v);
        Ok(prepend_intercept(rotated.view()))
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        let cr = CrSpline::new(self.knots.clone())?;
        let raw_d1 = cr.d1(x_new, axis);
        let rotated_d1 = raw_d1.dot(&self.centring).dot(&self.reparam_v);
        Ok(prepend_zero_column(rotated_d1.view()))
    }

    /// Build the rotated centred design + rotation V for a single CR
    /// smooth fit on column `x_col_idx`. Shared with `Additive` so a
    /// mixed `s(x, bs="cr")` / `s(z, bs="cs")` formula composes cleanly.
    pub(crate) fn fit_rotated(
        x: ArrayView2<f64>,
        x_col_idx: usize,
        k: usize,
        pc: Option<f64>,
        by: Option<usize>,
    ) -> Result<RotatedCrFit> {
        let x_col = x.column(x_col_idx);
        let cr = CrSpline::with_quantile_knots(x_col, k)?;
        let knots = cr.knots().to_owned();
        let x_view = x.slice(ndarray::s![.., x_col_idx..x_col_idx + 1]);
        let raw_design = cr.evaluate(x_view);
        let s_raw = cr.penalties().pop().unwrap();

        let x_inf = matrix_inf_norm(raw_design.view());
        let s_inf = matrix_inf_norm(s_raw.view());
        let rescale = if s_inf > 0.0 {
            (x_inf * x_inf) / s_inf
        } else {
            1.0
        };
        let s_raw_rescaled = &s_raw * rescale;

        let w = by.map(|b| x.column(b));
        let centring = constraint_map(cr, raw_design.view(), pc, w)?;
        let s_centred_rescaled = centring.t().dot(&s_raw_rescaled).dot(&centring);

        let (_, reparam_v) = descending_eigh(s_centred_rescaled.view())?;
        let mut rotated = raw_design.dot(&centring).dot(&reparam_v);
        // After the penalty rescale: mgcv sizes it on the unscaled basis.
        scale_rows(&mut rotated, w);
        let s_smooth = reparam_v.t().dot(&s_centred_rescaled).dot(&reparam_v);

        Ok(RotatedCrFit {
            knots,
            centring,
            reparam_v,
            rotated,
            s_smooth,
        })
    }
}

pub(crate) struct RotatedCrFit {
    pub knots: Array1<f64>,
    pub centring: Array2<f64>,
    pub reparam_v: Array2<f64>,
    pub rotated: Array2<f64>,
    pub s_smooth: Array2<f64>,
}

/// CR + sum-to-zero + `StableReparam` rotation + intercept.
/// `StableReparam` (mgcv `Sl.initial.repara` analog) diagonalises the
/// inner penalty for improved numerical conditioning. Predictions are
/// basis-invariant w.r.t. [`Cr`] up to FP.
pub struct CrStable {
    pub k: usize,
}

impl DesignStrategy for CrStable {
    fn prepare(&self, x: ArrayView2<f64>) -> Result<PreparedDesign> {
        let fit = CrStablePredictor::fit_rotated(x, 0, self.k, None, None)?;
        let x_design = prepend_intercept(fit.rotated.view());
        let p = x_design.ncols();
        let k_smooth = fit.rotated.ncols();

        let mut s_block = Array2::<f64>::zeros((p, p));
        for i in 0..k_smooth {
            for j in 0..k_smooth {
                s_block[[1 + i, 1 + j]] = fit.s_smooth[[i, j]];
            }
        }

        let (rank_s, log_pseudo_det_s) = rank_and_log_pseudo_det(s_block.view())?;
        let mp = p - rank_s;

        Ok(PreparedDesign {
            x_design,
            s_list: vec![s_block],
            rank_s_list: vec![rank_s],
            log_pseudo_det_s_list: vec![log_pseudo_det_s],
            mp,
            predictor: Predictor::CrStable(CrStablePredictor {
                knots: fit.knots,
                centring: fit.centring,
                reparam_v: fit.reparam_v,
            }),
        })
    }
}
