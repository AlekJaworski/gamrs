//! mgcv's `bs` B-spline basis with a derivative-based penalty
//! (`smooth.construct.bs.smooth.spec`, Wood 2017), the basis mgcv's `scasm`
//! builds its shape-constrained `sc` smooths on.
//!
//! `m = (m1, m2)`: `m1` is the spline order (3 = cubic, so B-splines of
//! order `m1 + 1`), `m2` the derivative order in the penalty
//! `∫ f^{(m2)}(x)² dx` over the interior knot range. Knots are evenly
//! spaced, `k + m1 + 1` of them, with the interior range stretched 0.1%
//! past the data on each side. Prediction outside the interior range is
//! linear in the value and slope at the nearer end, as mgcv does.

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::error::{GamrsError, Result};
use crate::traits::Basis;

/// A fitted `bs` basis: knots and orders.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct BsBasis {
    pub knots: Vec<f64>,
    /// `(m1, m2)` as in mgcv's `m = c(m1, m2)`.
    pub m: (usize, usize),
    /// Basis dimension `k`.
    pub k: usize,
}

impl BsBasis {
    /// Knots as `smooth.construct.bs.smooth.spec` places them from data.
    pub fn from_data(x: ArrayView1<f64>, k: usize, m: (usize, usize)) -> Result<Self> {
        let (m1, m2) = m;
        if m2 > m1 {
            return Err(GamrsError::InvalidParameter(format!(
                "bs basis: penalty derivative order {m2} exceeds spline order {m1}"
            )));
        }
        if k < m1 + 2 {
            return Err(GamrsError::InvalidParameter(format!(
                "bs basis: k = {k} too small for spline order {m1}"
            )));
        }
        if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
            return Err(GamrsError::InvalidParameter(
                "bs basis: covariate must be non-empty and finite".into(),
            ));
        }
        let nk = k - m1 + 1;
        let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = hi - lo;
        let xl = lo - range * 0.001;
        let xu = hi + range * 0.001;
        let dx = (xu - xl) / (nk - 1) as f64;
        let knots = r_seq(xl - dx * m1 as f64, xu + dx * m1 as f64, nk + 2 * m1);
        Ok(Self { knots, m, k })
    }

    /// Rebuild from stored knots (`k + m1 + 1` of them).
    pub fn from_knots(knots: Vec<f64>, m: (usize, usize)) -> Self {
        let k = knots.len() - m.0 - 1;
        Self { knots, m, k }
    }

    fn order(&self) -> usize {
        self.m.0 + 1
    }

    /// Interior knot range `[ll, ul]`: the data range the basis was built on.
    pub fn interior(&self) -> (f64, f64) {
        let m1 = self.m.0;
        (self.knots[m1], self.knots[self.knots.len() - m1 - 1])
    }

    /// Model matrix rows at `x` (n × k), linear extrapolation outside the
    /// interior range (mgcv's `Predict.matrix.pspline.smooth`).
    pub fn design(&self, x: ArrayView1<f64>) -> Array2<f64> {
        self.design_deriv(x, 0)
    }

    /// Rows of the `deriv`-th derivative of the basis. Outside the interior
    /// range the first derivative is the end slope and higher ones are 0.
    pub fn design_deriv(&self, x: ArrayView1<f64>, deriv: usize) -> Array2<f64> {
        let (ll, ul) = self.interior();
        let ord = self.order();
        let mut out = Array2::<f64>::zeros((x.len(), self.k));
        let ends = || {
            [
                basis_row(&self.knots, ll, ord, 0),
                basis_row(&self.knots, ll, ord, 1),
                basis_row(&self.knots, ul, ord, 0),
                basis_row(&self.knots, ul, ord, 1),
            ]
        };
        let mut end_rows: Option<[Vec<f64>; 4]> = None;
        for (i, &xi) in x.iter().enumerate() {
            let row = if xi >= ll && xi <= ul {
                basis_row(&self.knots, xi, ord, deriv)
            } else {
                let e = end_rows.get_or_insert_with(ends);
                let (v, s, d) = if xi < ll {
                    (&e[0], &e[1], xi - ll)
                } else {
                    (&e[2], &e[3], xi - ul)
                };
                match deriv {
                    0 => v.iter().zip(s).map(|(a, b)| a + d * b).collect(),
                    1 => s.clone(),
                    _ => vec![0.0; self.k],
                }
            };
            for (j, r) in row.into_iter().enumerate() {
                out[[i, j]] = r;
            }
        }
        out
    }

    /// Unscaled derivative penalty `S = Dᵀ W D` (k × k): `D` holds the
    /// `m2`-th derivative of the basis at `pord + 1` points per interior
    /// interval and `W` integrates the piecewise polynomial of degree
    /// `pord = m1 - m2` exactly, as mgcv's constructor does.
    pub fn penalty(&self) -> Array2<f64> {
        let (m1, m2) = self.m;
        let nk = self.k - m1 + 1;
        let k0: Vec<f64> = self.knots[m1..m1 + nk].to_vec();
        let h: Vec<f64> = k0.windows(2).map(|w| w[1] - w[0]).collect();
        let pord = m1 - m2;
        let ord = self.order();
        if pord == 0 {
            // Step-function integrand: one point per interval, weight h.
            let mut s = Array2::<f64>::zeros((self.k, self.k));
            for (q, hq) in h.iter().enumerate() {
                let mid = (k0[q] + k0[q + 1]) / 2.0;
                let d = basis_row(&self.knots, mid, ord, m2);
                add_outer(&mut s, &d, *hq);
            }
            return s;
        }
        let mut pts = vec![k0[0]];
        for hq in &h {
            for _ in 0..pord {
                let last = *pts.last().unwrap();
                pts.push(last + hq / pord as f64);
            }
        }
        let d: Vec<Vec<f64>> = pts
            .iter()
            .map(|&p| basis_row(&self.knots, p, ord, m2))
            .collect();
        let w1 = interval_weights(pord);
        let mut s = Array2::<f64>::zeros((self.k, self.k));
        for (q, hq) in h.iter().enumerate() {
            let base = q * pord;
            for a in 0..=pord {
                for b in 0..=pord {
                    let wab = w1[a][b] * hq / 2.0;
                    if wab == 0.0 {
                        continue;
                    }
                    for i in 0..self.k {
                        let dai = d[base + a][i];
                        if dai == 0.0 {
                            continue;
                        }
                        for j in 0..self.k {
                            s[[i, j]] += wab * dai * d[base + b][j];
                        }
                    }
                }
            }
        }
        s
    }
}

/// `W1 = Pᵀ H P` for exact integration over [-1, 1] of a degree-`pord`
/// polynomial through `pord + 1` equally spaced points.
fn interval_weights(pord: usize) -> Vec<Vec<f64>> {
    let np = pord + 1;
    let s: Vec<f64> = (0..np)
        .map(|i| -1.0 + 2.0 * i as f64 / pord as f64)
        .collect();
    let v = Array2::from_shape_fn((np, np), |(i, j)| s[i].powi(j as i32));
    let p = invert_small(&v);
    let hm = Array2::from_shape_fn((np, np), |(i, j)| {
        let e = i + j;
        if e % 2 == 0 {
            2.0 / (e + 1) as f64
        } else {
            0.0
        }
    });
    let w = p.t().dot(&hm).dot(&p);
    (0..np)
        .map(|i| (0..np).map(|j| w[[i, j]]).collect())
        .collect()
}

fn invert_small(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut m = a.clone();
    let mut inv = Array2::<f64>::eye(n);
    for c in 0..n {
        let piv = (c..n)
            .max_by(|&i, &j| m[[i, c]].abs().partial_cmp(&m[[j, c]].abs()).unwrap())
            .unwrap();
        if piv != c {
            for j in 0..n {
                m.swap([c, j], [piv, j]);
                inv.swap([c, j], [piv, j]);
            }
        }
        let d = m[[c, c]];
        for j in 0..n {
            m[[c, j]] /= d;
            inv[[c, j]] /= d;
        }
        for i in 0..n {
            if i != c {
                let f = m[[i, c]];
                for j in 0..n {
                    m[[i, j]] -= f * m[[c, j]];
                    inv[[i, j]] -= f * inv[[c, j]];
                }
            }
        }
    }
    inv
}

fn add_outer(s: &mut Array2<f64>, d: &[f64], w: f64) {
    for i in 0..d.len() {
        for j in 0..d.len() {
            s[[i, j]] += w * d[i] * d[j];
        }
    }
}

/// R's `seq(from, to, length.out = n)`: exact endpoints, interior points
/// `from + i * (to - from) / (n - 1)`.
pub(crate) fn r_seq(from: f64, to: f64, n: usize) -> Vec<f64> {
    let by = (to - from) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| from + i as f64 * by).collect();
    v[n - 1] = to;
    v
}

/// All `len(knots) - ord` B-splines of order `ord` (or their `deriv`-th
/// derivative) at `x`, for `x` inside `[knots[ord-1], knots[len-ord]]`; the
/// right end belongs to the last interval, as in `splines::splineDesign`.
pub(crate) fn basis_row(knots: &[f64], x: f64, ord: usize, deriv: usize) -> Vec<f64> {
    let nt = knots.len();
    let nb = nt - ord;
    if deriv >= ord {
        return vec![0.0; nb];
    }
    // Interval index mu with knots[mu] <= x < knots[mu+1], clamped to the
    // valid range so x at the right end uses the last interval.
    let lo = ord - 1;
    let hi = nt - ord - 1;
    let mut mu = lo;
    while mu < hi && x >= knots[mu + 1] {
        mu += 1;
    }
    // Order-1 indicators, then Cox-de Boor up to order ord - deriv.
    let base = ord - deriv;
    let mut b = vec![0.0; nt - 1];
    b[mu] = 1.0;
    for j in 2..=base {
        let mut nb_j = vec![0.0; nt - j];
        for i in 0..nt - j {
            let mut v = 0.0;
            let d1 = knots[i + j - 1] - knots[i];
            if d1 > 0.0 {
                v += (x - knots[i]) / d1 * b[i];
            }
            let d2 = knots[i + j] - knots[i + 1];
            if d2 > 0.0 {
                v += (knots[i + j] - x) / d2 * b[i + 1];
            }
            nb_j[i] = v;
        }
        b = nb_j;
    }
    // Differentiate up from order `base` to `ord`.
    for r in base + 1..=ord {
        let mut nb_r = vec![0.0; nt - r];
        for i in 0..nt - r {
            let mut v = 0.0;
            let d1 = knots[i + r - 1] - knots[i];
            if d1 > 0.0 {
                v += b[i] / d1;
            }
            let d2 = knots[i + r] - knots[i + 1];
            if d2 > 0.0 {
                v -= b[i + 1] / d2;
            }
            nb_r[i] = (r - 1) as f64 * v;
        }
        b = nb_r;
    }
    debug_assert_eq!(b.len(), nb);
    b
}

impl Basis for BsBasis {
    fn dim(&self) -> usize {
        self.k
    }

    fn input_dim(&self) -> usize {
        1
    }

    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64> {
        self.design(x.column(0))
    }

    fn d1(&self, x: ArrayView2<f64>, axis: usize) -> Array2<f64> {
        debug_assert_eq!(axis, 0, "BsBasis is univariate");
        self.design_deriv(x.column(0), 1)
    }

    fn penalties(&self) -> Vec<Array2<f64>> {
        vec![self.penalty()]
    }
}
