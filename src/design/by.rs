//! mgcv's numeric `by=` variable: the term contributes `w_i · f(x_i)`.
//!
//! The fit-time scaling happens where the term's basis is built (see
//! `CrPredictor::fit_centred`); [`ByPredictor`] wraps that term's own
//! predictor and applies the same scaling to new rows.

use ndarray::{s, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::Result;

use super::Predictor;

/// Multiply row `i` of `m` by `w[i]`; a no-op without a by-variable.
pub(crate) fn scale_rows(m: &mut Array2<f64>, w: Option<ArrayView1<f64>>) {
    if let Some(w) = w {
        *m *= &w.insert_axis(Axis(1));
    }
}

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct ByPredictor {
    /// The smooth's own predictor, invoked on the single column `col`.
    pub term: Box<Predictor>,
    pub col: usize,
    pub by: usize,
}

impl ByPredictor {
    /// `term` itself without a by-variable, so a plain smooth keeps the
    /// predictor (and persisted form) it always had.
    pub(crate) fn wrap(term: Predictor, col: usize, by: Option<usize>) -> Predictor {
        match by {
            Some(by) => Predictor::By(Self {
                term: Box::new(term),
                col,
                by,
            }),
            None => term,
        }
    }

    /// Reads columns `col` and `by` of the full `x_new`. The leading
    /// intercept column stays unscaled.
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut d = self.term.design(self.smooth_input(x_new))?;
        self.scale_smooth(&mut d, x_new);
        Ok(d)
    }

    /// `∂(w·f(x)) = w·f'(x)·∂x + f(x)·∂w`; `s(x, by=x)` takes both parts.
    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        let x_col = self.smooth_input(x_new);
        let mut d = self.term.design_deriv(x_col, 0)?;
        self.scale_smooth(&mut d, x_new);
        if axis != self.col {
            d.fill(0.0);
        }
        if axis == self.by {
            let f = self.term.design(x_col)?;
            let mut part = d.slice_mut(s![.., 1..]);
            part += &f.slice(s![.., 1..]);
        }
        Ok(d)
    }

    fn smooth_input<'a>(&self, x_new: ArrayView2<'a, f64>) -> ArrayView2<'a, f64> {
        x_new.slice_move(s![.., self.col..self.col + 1])
    }

    fn scale_smooth(&self, d: &mut Array2<f64>, x_new: ArrayView2<f64>) {
        let w = x_new.column(self.by).insert_axis(Axis(1));
        let mut smooth = d.slice_mut(s![.., 1..]);
        smooth *= &w;
    }
}
