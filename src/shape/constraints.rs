//! Linear constraints of a scasm model: inequality rows `A β ≥ b` from each
//! smooth's shape and one sum-to-zero equality row `C β = 0` per centred
//! smooth, both over the full coefficient vector (intercept first).
//!
//! scasm keeps the centring as an equality constraint in the fit instead of
//! absorbing it into the basis, so the shape rows act on the raw B-spline
//! coefficients: for a B-spline the function is increasing wherever its
//! coefficients are, so `β_{j+1} − β_j ≥ 0` is the increasing condition.

use ndarray::{Array1, Array2};

/// Monotone direction of a 1-D smooth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Increasing,
    Decreasing,
}

impl Shape {
    fn sign(self) -> f64 {
        match self {
            Shape::Increasing => 1.0,
            Shape::Decreasing => -1.0,
        }
    }
}

/// The shape rows of one `k`-coefficient smooth: `±(β_{j+1} − β_j) ≥ 0`
/// (`smooth.construct.sc.smooth.spec`, `xt = "m+"` / `"m-"`).
pub fn shape_rows(shape: Shape, k: usize) -> Array2<f64> {
    let s = shape.sign();
    let mut a = Array2::<f64>::zeros((k - 1, k));
    for j in 0..k - 1 {
        a[[j, j]] = -s;
        a[[j, j + 1]] = s;
    }
    a
}

/// All constraints of a model with `p` coefficients.
#[derive(Debug, Clone)]
pub struct Constraints {
    /// Inequality rows, `ain · β ≥ bin`.
    pub ain: Array2<f64>,
    pub bin: Array1<f64>,
    /// Equality rows, `c · β = 0`.
    pub c: Array2<f64>,
}

impl Constraints {
    pub fn empty(p: usize) -> Self {
        Self {
            ain: Array2::zeros((0, p)),
            bin: Array1::zeros(0),
            c: Array2::zeros((0, p)),
        }
    }

    /// Append a smooth occupying columns `first..first + rows.ncols()`:
    /// its inequality rows and, when given, its sum-to-zero row.
    pub fn add_smooth(&mut self, first: usize, rows: &Array2<f64>, centre: Option<&Array1<f64>>) {
        let p = self.ain.ncols();
        let k = rows.ncols();
        let mut a = Array2::<f64>::zeros((self.ain.nrows() + rows.nrows(), p));
        a.slice_mut(ndarray::s![..self.ain.nrows(), ..])
            .assign(&self.ain);
        a.slice_mut(ndarray::s![self.ain.nrows().., first..first + k])
            .assign(rows);
        let (a, b) = unique_rows(a, Array1::zeros(self.ain.nrows() + rows.nrows()));
        self.ain = a;
        self.bin = b;
        if let Some(cm) = centre {
            let mut c = Array2::<f64>::zeros((self.c.nrows() + 1, p));
            c.slice_mut(ndarray::s![..self.c.nrows(), ..])
                .assign(&self.c);
            c.slice_mut(ndarray::s![self.c.nrows(), first..first + k])
                .assign(cm);
            self.c = c;
        }
    }
}

/// mgcv's `ucon`: scale each row of `[A b]` so its first non-zero entry has
/// magnitude 1 and drop duplicates, keeping first occurrences in order.
pub fn unique_rows(a: Array2<f64>, b: Array1<f64>) -> (Array2<f64>, Array1<f64>) {
    let (n, p) = a.dim();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut r: Vec<f64> = a.row(i).to_vec();
        r.push(b[i]);
        if let Some(lead) = r.iter().find(|v| **v != 0.0).map(|v| v.abs()) {
            r.iter_mut().for_each(|v| *v /= lead);
        }
        if !rows.contains(&r) {
            rows.push(r);
        }
    }
    let m = rows.len();
    let mut ao = Array2::<f64>::zeros((m, p));
    let mut bo = Array1::<f64>::zeros(m);
    for (i, r) in rows.into_iter().enumerate() {
        for j in 0..p {
            ao[[i, j]] = r[j];
        }
        bo[i] = r[p];
    }
    (ao, bo)
}

/// A strictly feasible start for a monotone 1-D smooth block: successive
/// coefficients differ by exactly `ε^{1/4}` in the constrained direction
/// and the block satisfies its sum-to-zero row. This is the vertex mgcv's
/// phase-1 LP (`feasible`) returns for these constraints.
pub fn monotone_start(shape: Shape, centre: &Array1<f64>) -> Array1<f64> {
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
