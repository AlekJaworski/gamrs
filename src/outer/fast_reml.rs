//! mgcv's `fast.REML.fit`: the Newton loop `bam(method = "fREML")` runs over
//! `(ρ, log φ)`.
//!
//! Ported step for step, stopping rule included, because on a flat REML ridge
//! the reported λ is where this loop stops rather than an optimum.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

use crate::error::{GamrsError, Result};
use crate::traits::OuterFit;

/// One evaluation of the criterion: value, gradient and Hessian at θ, plus
/// the unpenalised residual sum of squares per observation that mgcv adds to
/// `|REML|` to scale its tolerances.
pub struct FastRemlProbe {
    pub value: f64,
    pub grad: Array1<f64>,
    pub hess: Array2<f64>,
    pub rss_per_obs: f64,
}

pub trait FastRemlScore {
    fn probe(&self, theta: &Array1<f64>) -> Result<FastRemlProbe>;
}

pub struct FastRemlNewton {
    pub conv_tol: f64,
    pub max_iter: usize,
    pub max_step: f64,
    pub max_halvings: usize,
}

impl Default for FastRemlNewton {
    fn default() -> Self {
        Self {
            conv_tol: f64::EPSILON.sqrt(),
            max_iter: 200,
            max_step: 5.0,
            max_halvings: 25,
        }
    }
}

impl FastRemlNewton {
    pub fn minimize<S: FastRemlScore>(&self, score: &S, theta0: Array1<f64>) -> Result<OuterFit> {
        let tol = self.conv_tol;
        let mut rho = theta0;
        let mut best = score.probe(&rho)?;
        let mut reml_scale = best.value.abs() + best.rss_per_obs;
        let mut active = unconverged_axes(&best, reml_scale * tol * 0.1);
        if !active.iter().any(|&a| a) {
            active.iter_mut().for_each(|a| *a = true);
        }
        let mut grad = best.grad.clone();
        let mut step_failed = false;
        let mut converged = false;
        let mut iter = 0;
        while iter < self.max_iter {
            iter += 1;
            let step = self.newton_step(&best, &active)?;
            let mut step = step;
            let mut trial = score.probe(&(&rho + &step))?;
            let mut halvings = 0;
            let mut not_moved = 0;
            while trial.value > best.value {
                if trial.value - best.value < tol * reml_scale {
                    not_moved += 1;
                } else {
                    not_moved = 0;
                }
                let moves = rho.iter().zip(step.iter()).any(|(&r, &s)| r + s != r);
                if halvings == self.max_halvings || !moves || not_moved > 3 {
                    step_failed = true;
                    break;
                }
                step.mapv_inplace(|s| s / 2.0);
                halvings += 1;
                trial = score.probe(&(&rho + &step))?;
            }
            if step_failed {
                break;
            }
            converged = true;
            grad = trial.grad.clone();
            active = unconverged_axes(&trial, reml_scale * tol * 0.1);
            if grad.iter().any(|g| g.abs() > reml_scale * tol) {
                converged = false;
            }
            if (best.value - trial.value).abs() > reml_scale * tol {
                if converged {
                    active.iter_mut().for_each(|a| *a = true);
                }
                converged = false;
            }
            best = trial;
            rho += &step;
            if converged {
                break;
            }
            reml_scale = best.value.abs() + best.rss_per_obs;
        }
        Ok(OuterFit {
            theta: rho,
            value: best.value,
            grad_norm: grad.iter().fold(0.0_f64, |m, g| m.max(g.abs())),
            iterations: iter,
            converged: converged || step_failed,
        })
    }

    /// Newton step on the unconverged axes: Hessian eigenvalues made positive
    /// and floored at `sqrt(eps)·max|eigenvalue|`, then capped at `max_step`.
    fn newton_step(&self, at: &FastRemlProbe, active: &[bool]) -> Result<Array1<f64>> {
        let idx: Vec<usize> = (0..active.len()).filter(|&i| active[i]).collect();
        let m = idx.len();
        let hess = Array2::from_shape_fn((m, m), |(a, b)| at.hess[[idx[a], idx[b]]]);
        let grad = Array1::from_shape_fn(m, |a| at.grad[idx[a]]);
        let (vals, vecs) = hess
            .eigh(UPLO::Lower)
            .map_err(|e| GamrsError::Linalg(format!("fast REML Newton step: {e}")))?;
        let mut vals = vals.mapv(f64::abs);
        let thresh = vals.iter().fold(0.0_f64, |m, v| m.max(*v)) * f64::EPSILON.sqrt();
        vals.mapv_inplace(|v| v.max(thresh));
        let mut uc = -vecs.dot(&(vecs.t().dot(&grad) / &vals));
        let ms = uc.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if ms > self.max_step {
            uc.mapv_inplace(|v| v * self.max_step / ms);
        }
        let mut step = Array1::<f64>::zeros(active.len());
        for (a, &i) in idx.iter().enumerate() {
            step[i] = uc[a];
        }
        Ok(step)
    }
}

fn unconverged_axes(at: &FastRemlProbe, cut: f64) -> Vec<bool> {
    (0..at.grad.len())
        .map(|i| at.grad[i].abs() > cut || at.hess[[i, i]].abs() > cut)
        .collect()
}
