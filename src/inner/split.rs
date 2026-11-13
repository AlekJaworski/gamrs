//! Products with a design matrix whose sparse columns (random-effect
//! indicators, `by`-scaled or not) are kept apart from its dense ones.
//!
//! `X'WX` costs `O(n·p²)` dense; with the sparse columns accumulated row by
//! row it costs `O(n·d²)` for the `d` dense columns, which for a factor with
//! hundreds of levels is the difference between seconds and milliseconds.
//! Results equal the dense products up to summation order.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::design::PenaltyBlock;

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
        let x_c = x.as_standard_layout();
        let rows = || x_c.as_slice().expect("standard layout").chunks_exact(p);
        let mut nnz = vec![0usize; p];
        for row in rows() {
            for (k, &v) in nnz.iter_mut().zip(row) {
                *k += (v != 0.0) as usize;
            }
        }
        let is_sparse: Vec<bool> = nnz.iter().map(|&k| 8 * k <= n).collect();
        let sparse: Vec<usize> = (0..p).filter(|&c| is_sparse[c]).collect();
        if sparse.len() < MIN_SPARSE_COLS {
            return None;
        }
        let dense: Vec<usize> = (0..p).filter(|&c| !is_sparse[c]).collect();
        let xd = x_c.select(Axis(1), &dense).as_standard_layout().into_owned();
        let total: usize = sparse.iter().map(|&c| nnz[c]).sum();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut cols = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        row_ptr.push(0);
        for row in rows() {
            for &c in &sparse {
                if row[c] != 0.0 {
                    cols.push(c);
                    vals.push(row[c]);
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

    fn xd_row(&self, r: usize) -> &[f64] {
        row(&self.xd, r)
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
            let wx = &self.x * &w.insert_axis(Axis(1));
            return self.x.t().dot(&wx);
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
        let mut pos = vec![0usize; p];
        for (a, &ca) in s.sparse.iter().enumerate() {
            pos[ca] = a;
        }
        let d = s.dense.len();
        let mut cross = vec![0.0; s.sparse.len() * d];
        for r in 0..self.x.nrows() {
            let range = s.row(r);
            let xr = s.xd_row(r);
            for t in range.clone() {
                let ca = s.cols[t];
                let f = w[r] * s.vals[t];
                for (o, &x) in cross[pos[ca] * d..][..d].iter_mut().zip(xr) {
                    *o += f * x;
                }
                for u in range.clone() {
                    out[[ca, s.cols[u]]] += f * s.vals[u];
                }
            }
        }
        for (a, &ca) in s.sparse.iter().enumerate() {
            for (j, &cj) in s.dense.iter().enumerate() {
                out[[ca, cj]] = cross[a * d + j];
                out[[cj, ca]] = cross[a * d + j];
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

    /// `diag(X · A⁻¹S_jA⁻¹ · X')` for each penalty `S_j`.
    pub(crate) fn penalty_leverages(
        &self,
        a_inv: ArrayView2<f64>,
        s_list: &[Array2<f64>],
        blocks: &[PenaltyBlock],
    ) -> Vec<Array1<f64>> {
        if self.split.is_some() {
            return s_list
                .iter()
                .zip(blocks)
                .map(|(s_j, b)| {
                    let cols = a_inv.slice(s![.., b.range.clone()]);
                    let m = times_block(cols, s_j, b).dot(&a_inv.slice(s![b.range.clone(), ..]));
                    self.leverage(m.view())
                })
                .collect();
        }
        let xa = self.x.dot(&a_inv);
        s_list
            .iter()
            .zip(blocks)
            .map(|(s_j, b)| {
                let xa_b = xa.slice(s![.., b.range.clone()]);
                (&times_block(xa_b, s_j, b) * &xa_b).sum_axis(Axis(1))
            })
            .collect()
    }

    /// Leverages `diag(X · M · X')` for a `(p, p)` matrix `M`.
    pub(crate) fn leverage(&self, m: ArrayView2<f64>) -> Array1<f64> {
        let Some(s) = &self.split else {
            let xm = self.x.dot(&m);
            return Array1::from_shape_fn(self.x.nrows(), |r| xm.row(r).dot(&self.x.row(r)));
        };
        let m_d = m.select(Axis(0), &s.dense);
        let m_cd = m.select(Axis(1), &s.dense).as_standard_layout().into_owned();
        let m_dc = m_d.t().as_standard_layout().into_owned();
        let xm = s.xd.dot(&m_d.select(Axis(1), &s.dense));
        Array1::from_shape_fn(self.x.nrows(), |r| {
            let xr = s.xd_row(r);
            let mut h = dot(xr, row(&xm, r));
            for t in s.row(r) {
                let (c, v) = (s.cols[t], s.vals[t]);
                h += v * (dot(row(&m_cd, c), xr) + dot(row(&m_dc, c), xr));
                for u in s.row(r) {
                    h += v * s.vals[u] * m[[c, s.cols[u]]];
                }
            }
            h
        })
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Row `r` of a standard-layout matrix.
fn row(a: &Array2<f64>, r: usize) -> &[f64] {
    let k = a.ncols();
    &a.as_slice().expect("standard layout")[r * k..][..k]
}

/// `Y · S[block, block]` for the `block` columns `Y` of a penalty `S`.
fn times_block(y: ArrayView2<f64>, s: &Array2<f64>, b: &PenaltyBlock) -> Array2<f64> {
    let s_b = s.slice(s![b.range.clone(), b.range.clone()]);
    if b.diagonal {
        &y * &s_b.diag()
    } else {
        y.dot(&s_b)
    }
}

/// `A · B` without the zero products of a `B` that is dense only in a few
/// rows and columns, as `X'WX` is outside its random-effect block.
pub(crate) fn dot_sparse_cols(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (p, q) = b.dim();
    let a_t = a.t().as_standard_layout().into_owned();
    let b_t = b.t().as_standard_layout().into_owned();
    let row_nnz = b.rows().into_iter().map(|r| r.iter().filter(|&&v| v != 0.0).count());
    let dense_rows: Vec<usize> = row_nnz.enumerate().filter(|&(_, k)| 4 * k > q).map(|(r, _)| r).collect();
    let other_rows: Vec<usize> = (0..p).filter(|r| !dense_rows.contains(r)).collect();
    let mut out_t = b_t.select(Axis(1), &dense_rows).dot(&a_t.select(Axis(0), &dense_rows));
    let mut dense_cols = Vec::new();
    for (c, b_col) in b_t.rows().into_iter().enumerate() {
        let nz: Vec<usize> = other_rows.iter().copied().filter(|&r| b_col[r] != 0.0).collect();
        if 4 * nz.len() > p {
            dense_cols.push(c);
            continue;
        }
        let mut o = out_t.row_mut(c);
        for r in nz {
            o.scaled_add(b_col[r], &a_t.row(r));
        }
    }
    if !dense_cols.is_empty() {
        let rest = b_t
            .select(Axis(0), &dense_cols)
            .select(Axis(1), &other_rows)
            .dot(&a_t.select(Axis(0), &other_rows));
        for (j, &c) in dense_cols.iter().enumerate() {
            let mut o = out_t.row_mut(c);
            o += &rest.row(j);
        }
    }
    out_t.reversed_axes().as_standard_layout().into_owned()
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
        let lev = x.dot(&m).dot(&x.t()).diag().to_owned();
        assert!((&xs.leverage(m.view()) - &lev).iter().all(|d| d.abs() < 1e-11));
        let mut s_re = Array2::<f64>::zeros((p, p));
        let mut s_sm = Array2::<f64>::zeros((p, p));
        for c in q..p {
            s_re[[c, c]] = 1.0;
        }
        for a in 1..q {
            for b in 1..q {
                s_sm[[a, b]] = 1.0 / (1 + a + b) as f64;
            }
        }
        let a_sym = &m + &m.t();
        let s_list = [s_re, s_sm];
        let blocks: Vec<PenaltyBlock> = s_list.iter().map(|s| PenaltyBlock::of(s.view())).collect();
        let dense = SplitDesign { x: x.view(), split: None };
        for (lev_s, lev_d) in xs
            .penalty_leverages(a_sym.view(), &s_list, &blocks)
            .iter()
            .zip(dense.penalty_leverages(a_sym.view(), &s_list, &blocks))
        {
            assert!((lev_s - &lev_d).iter().all(|d| d.abs() < 1e-10 * (1.0 + lev_d[0].abs())));
        }
        let g = xs.xtwx(w.view());
        assert!(max_abs(&dot_sparse_cols(m.view(), g.view()), &m.dot(&g)) < 1e-11);
    }
}
