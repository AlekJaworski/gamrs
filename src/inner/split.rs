//! Products with a design matrix whose sparse columns (random-effect
//! indicators, `by`-scaled or not) are kept apart from its dense ones.
//!
//! `X'WX` costs `O(n·p²)` dense; with the sparse columns accumulated row by
//! row it costs `O(n·d²)` for the `d` dense columns, which for a factor with
//! hundreds of levels is the difference between seconds and milliseconds.
//! Results equal the dense products up to summation order.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Below this many sparse columns the dense BLAS products are as fast.
const MIN_SPARSE_COLS: usize = 32;

struct ColumnSplit {
    dense: Vec<usize>,
    sparse: Vec<usize>,
    /// `X[:, dense]`, contiguous.
    xd: Array2<f64>,
    /// CSR over the sparse columns: row `r` holds `cols/vals[row_ptr[r]..row_ptr[r+1]]`.
    row_ptr: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl ColumnSplit {
    fn of(x: ArrayView2<f64>) -> Option<Self> {
        let (n, p) = x.dim();
        if p < MIN_SPARSE_COLS {
            return None;
        }
        let mut nnz = vec![0usize; p];
        for row in x.rows() {
            for (c, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    nnz[c] += 1;
                }
            }
        }
        let is_sparse: Vec<bool> = nnz.iter().map(|&k| 8 * k <= n).collect();
        let sparse: Vec<usize> = (0..p).filter(|&c| is_sparse[c]).collect();
        if sparse.len() < MIN_SPARSE_COLS {
            return None;
        }
        let dense: Vec<usize> = (0..p).filter(|&c| !is_sparse[c]).collect();
        let xd = x.select(Axis(1), &dense);
        let total: usize = sparse.iter().map(|&c| nnz[c]).sum();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut cols = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        row_ptr.push(0);
        for row in x.rows() {
            for (c, &v) in row.iter().enumerate() {
                if is_sparse[c] && v != 0.0 {
                    cols.push(c);
                    vals.push(v);
                }
            }
            row_ptr.push(cols.len());
        }
        Some(Self {
            dense,
            sparse,
            xd,
            row_ptr,
            cols,
            vals,
        })
    }

    fn row(&self, r: usize) -> std::ops::Range<usize> {
        self.row_ptr[r]..self.row_ptr[r + 1]
    }
}

/// A borrowed design matrix plus, when it pays, its dense/sparse column split.
pub(crate) struct SplitDesign<'a> {
    x: ArrayView2<'a, f64>,
    split: Option<ColumnSplit>,
}

impl<'a> SplitDesign<'a> {
    pub(crate) fn new(x: ArrayView2<'a, f64>) -> Self {
        Self {
            x,
            split: ColumnSplit::of(x),
        }
    }

    /// `X' diag(w) X`.
    pub(crate) fn xtwx(&self, w: ArrayView1<f64>) -> Array2<f64> {
        let Some(s) = &self.split else {
            return super::weighted_xt_view(self.x, w).dot(&self.x);
        };
        let p = self.x.ncols();
        let mut out = Array2::<f64>::zeros((p, p));
        let wxd = &s.xd * &w.insert_axis(Axis(1));
        let gd = s.xd.t().dot(&wxd);
        for (a, &ca) in s.dense.iter().enumerate() {
            for (b, &cb) in s.dense.iter().enumerate() {
                out[[ca, cb]] = gd[[a, b]];
            }
        }
        for r in 0..self.x.nrows() {
            let range = s.row(r);
            if range.is_empty() {
                continue;
            }
            let xr = s.xd.row(r);
            for t in range.clone() {
                let ca = s.cols[t];
                let f = w[r] * s.vals[t];
                for (j, &cj) in s.dense.iter().enumerate() {
                    out[[ca, cj]] += f * xr[j];
                }
                for u in range.clone() {
                    out[[ca, s.cols[u]]] += f * s.vals[u];
                }
            }
        }
        for &ca in &s.sparse {
            for &cj in &s.dense {
                out[[cj, ca]] = out[[ca, cj]];
            }
        }
        out
    }

    /// `X · v`.
    pub(crate) fn dot_vec(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let Some(s) = &self.split else {
            return self.x.dot(&v);
        };
        let mut out = s.xd.dot(&v.select(Axis(0), &s.dense));
        for (r, o) in out.iter_mut().enumerate() {
            for t in s.row(r) {
                *o += s.vals[t] * v[s.cols[t]];
            }
        }
        out
    }

    /// `X' · v`.
    pub(crate) fn t_dot_vec(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let Some(s) = &self.split else {
            return self.x.t().dot(&v);
        };
        let mut out = Array1::<f64>::zeros(self.x.ncols());
        let d = s.xd.t().dot(&v);
        for (j, &cj) in s.dense.iter().enumerate() {
            out[cj] = d[j];
        }
        for (r, &vr) in v.iter().enumerate() {
            for t in s.row(r) {
                out[s.cols[t]] += s.vals[t] * vr;
            }
        }
        out
    }

    /// `X · M` for a `(p, k)` matrix `M`.
    pub(crate) fn dot(&self, m: ArrayView2<f64>) -> Array2<f64> {
        let Some(s) = &self.split else {
            return self.x.dot(&m);
        };
        let mut out = s.xd.dot(&m.select(Axis(0), &s.dense));
        for (r, mut o) in out.rows_mut().into_iter().enumerate() {
            for t in s.row(r) {
                o.scaled_add(s.vals[t], &m.row(s.cols[t]));
            }
        }
        out
    }

    /// Leverages `diag(X · M · X')` for a `(p, p)` matrix `M`.
    pub(crate) fn leverage(&self, m: ArrayView2<f64>) -> Array1<f64> {
        let Some(s) = &self.split else {
            let xm = self.x.dot(&m);
            return (&self.x * &xm).sum_axis(Axis(1));
        };
        let m_d = m.select(Axis(0), &s.dense);
        let m_cd = m.select(Axis(1), &s.dense);
        let m_dc = m_d.t().as_standard_layout().into_owned();
        let xm = s.xd.dot(&m_d.select(Axis(1), &s.dense));
        Array1::from_shape_fn(self.x.nrows(), |r| {
            let xr = s.xd.row(r);
            let mut h = xr.dot(&xm.row(r));
            for t in s.row(r) {
                let (c, v) = (s.cols[t], s.vals[t]);
                h += v * (m_cd.row(c).dot(&xr) + m_dc.row(c).dot(&xr));
                for u in s.row(r) {
                    h += v * s.vals[u] * m[[c, s.cols[u]]];
                }
            }
            h
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    #[test]
    fn split_products_match_dense() {
        let (n, levels, q) = (600, 60, 5);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut u = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let p = q + levels;
        let mut x = Array2::<f64>::zeros((n, p));
        for r in 0..n {
            for c in 0..q {
                x[[r, c]] = u() - 0.5;
            }
            // by-scaled indicator; some rows carry none
            if u() > 0.1 {
                x[[r, q + ((u() * levels as f64) as usize).min(levels - 1)]] = 1.0 + u();
            }
        }
        let w = Array1::from_shape_fn(n, |_| u() - 0.2);
        let v = Array1::from_shape_fn(n, |_| u());
        let b = Array1::from_shape_fn(p, |_| u());
        let m = Array2::from_shape_fn((p, p), |_| u());
        let xs = SplitDesign::new(x.view());
        assert!(xs.split.is_some());

        let wx = &x * &w.view().insert_axis(Axis(1));
        assert!(max_abs(&xs.xtwx(w.view()), &x.t().dot(&wx)) < 1e-12);
        assert!((&xs.dot_vec(b.view()) - &x.dot(&b)).iter().all(|d| d.abs() < 1e-12));
        assert!((&xs.t_dot_vec(v.view()) - &x.t().dot(&v)).iter().all(|d| d.abs() < 1e-12));
        assert!(max_abs(&xs.dot(m.view()), &x.dot(&m)) < 1e-12);
        let lev = x.dot(&m).dot(&x.t()).diag().to_owned();
        assert!((&xs.leverage(m.view()) - &lev).iter().all(|d| d.abs() < 1e-11));
    }
}
