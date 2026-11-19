//! The Gaussian REML criterion `bam(method = "fREML")` optimises: mgcv's
//! `Sl.fit`, with `log φ` a free coordinate rather than profiled out.
//!
//! θ = `(ρ_1, …, ρ_m, log φ)`. With `H = X'WX + Σ λ_j S_j` and
//! `Dp = ‖√W(y − Xβ̂)‖² + Σ λ_j β̂'S_jβ̂`,
//!
//! `V = Dp/(2φ) + (n − Mp)/2·log(2πφ) + ½log|H| − ½log|Σ λ_j S_j|+`,
//!
//! differentiated exactly through β̂(ρ). Unlike mgcv `gam`'s REML there is no
//! prior-weight term `Σ log w_i / 2`.
//!
//! Works, like `Sl.initial.repara`, in a basis where each penalty is the
//! identity on its range. There `λ_j tr(H⁻¹S_j) − rank_j = −tr_j(H⁻¹X'WX)`,
//! so the gradient on a flat ridge (λ → ∞) is computed without the
//! cancellation that otherwise decides where the Newton loop stops.

use std::ops::Range;

use ndarray::{s, Array1, Array2, ArrayView1};
use ndarray_linalg::{DeterminantC, Eigh, FactorizeC, InverseC, UPLO};

use crate::error::{GamrsError, Result};
use crate::inner::DesignMatrix;
use crate::outer::fast_reml::{FastRemlProbe, FastRemlScore};
use crate::traits::{CoordsKind, ScoreDerivatives};

pub struct FastRemlGaussianScore {
    /// `T` with `β = T β̃`: block-diagonal, `U_j diag(d_j^{-1/2}, 1)` per penalty.
    t: Array2<f64>,
    xtwx: Array2<f64>,
    xtwy: Array1<f64>,
    ytwy: f64,
    /// Range columns of each penalty in the reparametrised basis.
    ranges: Vec<Range<usize>>,
    /// All columns of each penalised term.
    blocks: Vec<Range<usize>>,
    mp: usize,
    n: usize,
    /// A known scale takes `log φ` out of θ.
    known_scale: Option<f64>,
}

/// The fit at one θ, in the original coefficient basis.
pub struct FastRemlFit {
    pub beta: Array1<f64>,
    pub h_inv: Array2<f64>,
    pub edf_total: f64,
    pub edf_per_term: Array1<f64>,
    pub rss: f64,
}

struct Solve {
    beta: Array1<f64>,
    h_inv: Array2<f64>,
    /// `H⁻¹X'WX`.
    f: Array2<f64>,
    log_det_h: f64,
    rss: f64,
}

impl FastRemlGaussianScore {
    /// Each `s_list[j]` must have one block of rows/columns the others do not
    /// touch (one penalty per term), with `rank_s_list[j]` positive eigenvalues.
    pub fn new(
        x_design: &DesignMatrix,
        y: ArrayView1<f64>,
        weights: Option<ArrayView1<f64>>,
        s_list: &[Array2<f64>],
        rank_s_list: &[usize],
        mp: usize,
    ) -> Result<Self> {
        let n = x_design.nrows();
        let p = x_design.ncols();
        let w = weights.map_or_else(|| Array1::ones(n), |w| w.to_owned());
        let xtwx = x_design.split().xtwx(w.view());
        let wy = &w * &y;
        let xtwy = x_design.t().dot(&wy);
        let ytwy = wy.dot(&y);

        let mut t = Array2::<f64>::eye(p);
        let mut ranges = Vec::with_capacity(s_list.len());
        let mut blocks = Vec::with_capacity(s_list.len());
        let mut covered = vec![false; p];
        for (s_j, &rank) in s_list.iter().zip(rank_s_list) {
            let rows: Vec<usize> = (0..p)
                .filter(|&r| s_j.row(r).iter().any(|&v| v != 0.0))
                .collect();
            let (first, last) = match (rows.first(), rows.last()) {
                (Some(&a), Some(&b)) => (a, b),
                _ => return Err(unsupported()),
            };
            if (first..=last).any(|r| covered[r]) {
                return Err(unsupported());
            }
            (first..=last).for_each(|r| covered[r] = true);
            let block = s_j.slice(s![first..=last, first..=last]);
            let (vals, vecs) = block
                .eigh(UPLO::Lower)
                .map_err(|e| GamrsError::Linalg(format!("bam penalty reparam: {e}")))?;
            let k = vals.len();
            // Descending eigenvalues: the range columns come first.
            for (out, i) in (0..k).rev().enumerate() {
                let scale = if out < rank {
                    vals[i].sqrt().recip()
                } else {
                    1.0
                };
                for r in 0..k {
                    t[[first + r, first + out]] = vecs[[r, i]] * scale;
                }
            }
            ranges.push(first..first + rank);
            blocks.push(first..last + 1);
        }
        let xtwx = t.t().dot(&xtwx).dot(&t);
        let xtwy = t.t().dot(&xtwy);
        Ok(Self {
            t,
            xtwx,
            xtwy,
            ytwy,
            ranges,
            blocks,
            mp,
            n,
            known_scale: None,
        })
    }

    /// Hold φ fixed (mgcv's `phi.fixed`): θ is then `ρ` alone.
    pub fn with_known_scale(mut self, phi: Option<f64>) -> Self {
        self.known_scale = phi;
        self
    }

    fn solve(&self, rho: ArrayView1<f64>) -> Result<Solve> {
        let p = self.xtwx.nrows();
        let mut h = self.xtwx.clone();
        for (range, &r) in self.ranges.iter().zip(rho.iter()) {
            let lambda = r.exp();
            for i in range.clone() {
                h[[i, i]] += lambda;
            }
        }
        let d: Array1<f64> = h.diag().mapv(|v| v.sqrt().recip());
        let scaled = Array2::from_shape_fn((p, p), |(i, j)| h[[i, j]] * d[i] * d[j]);
        let chol = scaled
            .factorizec(UPLO::Lower)
            .map_err(|e| GamrsError::Linalg(format!("bam fREML Cholesky: {e}")))?;
        let log_det_h = chol.ln_detc() - 2.0 * d.mapv(f64::ln).sum();
        let inv = chol
            .invc()
            .map_err(|e| GamrsError::Linalg(format!("bam fREML inverse: {e}")))?;
        let h_inv = Array2::from_shape_fn((p, p), |(i, j)| inv[[i, j]] * d[i] * d[j]);
        let beta = h_inv.dot(&self.xtwy);
        let f = h_inv.dot(&self.xtwx);
        let rss = self.ytwy - 2.0 * beta.dot(&self.xtwy) + beta.dot(&self.xtwx.dot(&beta));
        Ok(Solve {
            beta,
            h_inv,
            f,
            log_det_h,
            rss,
        })
    }

    /// `Sl.fit`'s `reml.pen`: the REML terms other than the fit to the
    /// working data, which `bam` adds to the family's AIC/2 to report REML.
    pub fn reml_penalty(&self, rho: ArrayView1<f64>, phi: f64) -> Result<f64> {
        let sol = self.solve(rho)?;
        let mut bsb = 0.0;
        let mut rank_rho = 0.0;
        for (range, &r) in self.ranges.iter().zip(rho.iter()) {
            bsb += r.exp() * sol.beta.slice(s![range.clone()]).mapv(|b| b * b).sum();
            rank_rho += range.len() as f64 * r;
        }
        let mp = self.mp as f64;
        Ok(
            (bsb / phi - mp * (2.0 * std::f64::consts::PI * phi).ln() + sol.log_det_h - rank_rho)
                / 2.0,
        )
    }

    /// β̂, `H⁻¹`, edf and the RSS at `rho`, mapped back to the design basis.
    pub fn fit_at(&self, rho: ArrayView1<f64>) -> Result<FastRemlFit> {
        let sol = self.solve(rho)?;
        let edf_per_term = self
            .blocks
            .iter()
            .map(|b| b.clone().map(|i| sol.f[[i, i]]).sum())
            .collect();
        Ok(FastRemlFit {
            beta: self.t.dot(&sol.beta),
            h_inv: self.t.dot(&sol.h_inv).dot(&self.t.t()),
            edf_total: sol.f.diag().sum(),
            edf_per_term,
            rss: sol.rss,
        })
    }
}

fn unsupported() -> GamrsError {
    GamrsError::InvalidParameter(
        "bam fREML supports one penalty per term (cr, re, parametric); \
         tensor products are not implemented yet"
            .into(),
    )
}

fn block(m: &Array2<f64>, a: &Range<usize>, b: &Range<usize>) -> Array2<f64> {
    m.slice(s![a.clone(), b.clone()]).to_owned()
}

impl FastRemlScore for FastRemlGaussianScore {
    fn probe(&self, theta: &Array1<f64>) -> Result<FastRemlProbe> {
        let m = self.ranges.len();
        let rho = theta.slice(s![..m]);
        let phi = self.known_scale.unwrap_or_else(|| theta[m].exp());
        let sol = self.solve(rho)?;
        let lambda: Vec<f64> = rho.iter().map(|r| r.exp()).collect();

        let beta_r: Vec<Array1<f64>> = self
            .ranges
            .iter()
            .map(|r| sol.beta.slice(s![r.clone()]).to_owned())
            .collect();
        let bsb: Vec<f64> = beta_r.iter().map(|b| b.dot(b)).collect();
        let dp = sol.rss + (0..m).map(|j| lambda[j] * bsb[j]).sum::<f64>();
        let n_mp = self.n as f64 - self.mp as f64;
        let rank_rho: f64 = (0..m).map(|j| self.ranges[j].len() as f64 * rho[j]).sum();
        let value = dp / (2.0 * phi)
            + 0.5 * n_mp * (2.0 * std::f64::consts::PI * phi).ln()
            + 0.5 * sol.log_det_h
            - 0.5 * rank_rho;

        let mut grad = Array1::<f64>::zeros(m + 1);
        let mut hess = Array2::<f64>::zeros((m + 1, m + 1));
        for j in 0..m {
            let rj = &self.ranges[j];
            let f_jj = block(&sol.f, rj, rj);
            grad[j] = 0.5 * (lambda[j] * bsb[j] / phi - f_jj.diag().sum());
            for k in j..m {
                let rk = &self.ranges[k];
                // G = λ H⁻¹ on a range block; its complement is F there.
                let g_jk = block(&sol.h_inv, rj, rk) * lambda[k];
                let pen = 2.0 * lambda[j] * beta_r[j].dot(&g_jk.dot(&beta_r[k])) / phi;
                let h = if j == k {
                    lambda[j] * bsb[j] / phi - pen + (&g_jk * &f_jj.t()).sum()
                } else {
                    let f_kj = block(&sol.f, rk, rj);
                    -pen - (&block(&sol.f, rj, rk) * &f_kj.t()).sum()
                };
                hess[[j, k]] = 0.5 * h;
                hess[[k, j]] = 0.5 * h;
            }
            hess[[j, m]] = -lambda[j] * bsb[j] / (2.0 * phi);
            hess[[m, j]] = hess[[j, m]];
        }
        grad[m] = 0.5 * (n_mp - dp / phi);
        hess[[m, m]] = dp / (2.0 * phi);

        if self.known_scale.is_some() {
            grad = grad.slice(s![..m]).to_owned();
            hess = hess.slice(s![..m, ..m]).to_owned();
        }
        Ok(FastRemlProbe {
            value,
            grad,
            hess,
            rss_per_obs: sol.rss / self.n as f64,
        })
    }
}

impl ScoreDerivatives for FastRemlGaussianScore {
    fn dim(&self) -> usize {
        self.ranges.len() + usize::from(self.known_scale.is_none())
    }

    fn coords(&self) -> CoordsKind {
        CoordsKind::Reparametrised("Sl.initial.repara")
    }

    fn value(&self, theta: &Array1<f64>) -> Result<f64> {
        Ok(self.probe(theta)?.value)
    }

    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        let p = self.probe(theta)?;
        Ok((p.value, p.grad))
    }

    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let p = self.probe(theta)?;
        Ok((p.value, p.grad, p.hess))
    }
}
