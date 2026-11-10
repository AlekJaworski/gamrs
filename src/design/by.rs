//! mgcv's numeric `by=` variable: the term contributes `w_i · f(x_i)`.
//!
//! The fit-time scaling happens where the term's basis is built (see
//! `CrPredictor::fit_centred`, `TensorTermFit::build`,
//! `RePredictor::fit_raw`); [`ByPredictor`]
//! wraps that term's own predictor and applies the same scaling to new rows.

use ndarray::{s, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::Result;

use super::Predictor;

/// mgcv's `sd(by) > mean(by) * .Machine$double.eps * 1000`, on `|mean|`:
/// with the signed mean a constant negative `w` would count as varying and
/// leave a constant inside the smooth, aliased with the intercept.
pub(crate) fn varies(w: ArrayView1<f64>) -> bool {
    let n = w.len() as f64;
    let mean = w.sum() / n;
    let ss: f64 = w.iter().map(|v| (v - mean).powi(2)).sum();
    (ss / (n - 1.0)).sqrt() > mean.abs() * f64::EPSILON * 1000.0
}

/// Multiply row `i` of `m` by `w[i]`; a no-op without a by-variable.
pub(crate) fn scale_rows(m: &mut Array2<f64>, w: Option<ArrayView1<f64>>) {
    if let Some(w) = w {
        *m *= &w.insert_axis(Axis(1));
    }
}

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct ByPredictor {
    /// The smooth's own predictor: a univariate one is invoked on the
    /// single column `col`, a tensor on the full `x_new`.
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

    /// Reads the term's own columns and `by` of the full `x_new`. The
    /// leading intercept column stays unscaled.
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut d = self.term.design(self.smooth_input(x_new))?;
        self.scale_smooth(&mut d, x_new);
        Ok(d)
    }

    /// `∂(w·f) = w·∂f + f·∂w`; `s(x, by=x)` takes both parts.
    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        let input = self.smooth_input(x_new);
        let mut d = if self.reads_full_x() {
            self.term.design_deriv(input, axis)?
        } else {
            let mut d = self.term.design_deriv(input, 0)?;
            if axis != self.col {
                d.fill(0.0);
            }
            d
        };
        self.scale_smooth(&mut d, x_new);
        if axis == self.by {
            let f = self.term.design(input)?;
            let mut part = d.slice_mut(s![.., 1..]);
            part += &f.slice(s![.., 1..]);
        }
        Ok(d)
    }

    fn reads_full_x(&self) -> bool {
        matches!(*self.term, Predictor::Tensor(_))
    }

    fn smooth_input<'a>(&self, x_new: ArrayView2<'a, f64>) -> ArrayView2<'a, f64> {
        if self.reads_full_x() {
            return x_new;
        }
        x_new.slice_move(s![.., self.col..self.col + 1])
    }

    fn scale_smooth(&self, d: &mut Array2<f64>, x_new: ArrayView2<f64>) {
        let w = x_new.column(self.by).insert_axis(Axis(1));
        let mut smooth = d.slice_mut(s![.., 1..]);
        smooth *= &w;
    }
}
