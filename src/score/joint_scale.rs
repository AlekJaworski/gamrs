//! The Gaussian REML score over `(ρ, τ = log φ)` jointly, the parameters
//! mgcv's outer Newton searches for a REML fit with unknown scale
//! (`estimate.gam`'s `scale.as.sp`).
//!
//! gamrs's Gaussian score profiles φ out: `φ̂(ρ) = Dp/(n − Mp)`. The optimum
//! over ρ is the same either way, but the Newton steps are not, and REML over
//! several smooths can have more than one local minimum. Searching the same
//! function from the same start (`initial.spg`, and `τ₀ = log(null.scale/10)`)
//! lands where mgcv lands; the profiled search from that start can take
//! another basin (`tests/parity_gaussian_path.rs`).
//!
//! The joint score is the profiled one with `φ̂` swapped for `e^τ`. Writing
//! `c = (n − Mp)/2`, `d_j = λ_j β'S_jβ` and `Dp = D + Σ d_j`:
//!
//! ```text
//! V(ρ, τ)  = V_p(ρ) + ½Dp·(e^{−τ} − 1/φ̂) + c·(τ − ln φ̂)
//! ∂V/∂ρ_j  = g_p,j + ½d_j·(e^{−τ} − 1/φ̂)
//! ∂V/∂τ    = c − ½Dp·e^{−τ}
//! ∂²V/∂ρ²  = H_p + ½(e^{−τ} − 1/φ̂)·∂d/∂ρ + d dᵀ/(2φ̂²(n − Mp))
//! ∂²V/∂ρ∂τ = −½d·e^{−τ}
//! ∂²V/∂τ²  = ½Dp·e^{−τ}
//! ```
//!
//! The last `H_ρρ` term removes the profile's `∂φ̂/∂ρ` chain
//! (`∂φ̂/∂ρ_i = d_i/(n − Mp)`), which the joint score does not have.

use ndarray::{s, Array1, Array2};

use super::envelope::UNPHYSICAL_SCORE;
use crate::error::Result;
use crate::fit::gaussian::GaussianScore;
use crate::inner::LinearSolver;
use crate::traits::{CoordsKind, ScoreDerivatives};

/// [`GaussianScore`] over `θ = (ρ, τ)`; see the module docs.
pub(crate) struct JointScaleScore<'a, S: LinearSolver> {
    profiled: &'a GaussianScore<S>,
    /// `n − Mp`.
    n_minus_mp: f64,
}

impl<'a, S: LinearSolver> JointScaleScore<'a, S> {
    pub(crate) fn new(profiled: &'a GaussianScore<S>, n: usize, mp: usize) -> Self {
        Self {
            profiled,
            n_minus_mp: n as f64 - mp as f64,
        }
    }

    fn split(theta: &Array1<f64>) -> (Array1<f64>, f64) {
        let m = theta.len() - 1;
        (theta.slice(s![..m]).to_owned(), theta[m])
    }

    fn joint_value(&self, v_p: f64, dp: f64, tau: f64) -> f64 {
        let phi_hat = dp / self.n_minus_mp;
        v_p + 0.5 * dp * ((-tau).exp() - 1.0 / phi_hat)
            + 0.5 * self.n_minus_mp * (tau - phi_hat.ln())
    }
}

impl<S: LinearSolver> ScoreDerivatives for JointScaleScore<'_, S> {
    fn dim(&self) -> usize {
        self.profiled.dim() + 1
    }

    fn coords(&self) -> CoordsKind {
        self.profiled.coords()
    }

    fn value(&self, theta: &Array1<f64>) -> Result<f64> {
        let (rho, tau) = Self::split(theta);
        let (v_p, dp) = self.profiled.value_with_dp(&rho)?;
        if v_p >= UNPHYSICAL_SCORE || dp <= 0.0 {
            return Ok(UNPHYSICAL_SCORE);
        }
        Ok(self.joint_value(v_p, dp, tau))
    }

    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        let (v, g, _) = self.value_grad_hess(theta)?;
        Ok((v, g))
    }

    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let (rho, tau) = Self::split(theta);
        let m = rho.len();
        let (v_p, g_p, h_p, parts) = self.profiled.value_grad_hess_with_scale_parts(&rho)?;
        let mut g = Array1::<f64>::zeros(m + 1);
        let mut h = Array2::<f64>::zeros((m + 1, m + 1));
        if v_p >= UNPHYSICAL_SCORE || parts.dp <= 0.0 {
            return Ok((UNPHYSICAL_SCORE, g, h));
        }
        let dp = parts.dp;
        let phi_hat = dp / self.n_minus_mp;
        let e = (-tau).exp();
        let shift = e - 1.0 / phi_hat;
        let unchain = 1.0 / (2.0 * phi_hat * phi_hat * self.n_minus_mp);
        for j in 0..m {
            g[j] = g_p[j] + 0.5 * parts.d[j] * shift;
            for i in 0..m {
                h[[i, j]] = h_p[[i, j]]
                    + 0.5 * shift * parts.d2[[i, j]]
                    + unchain * parts.d[i] * parts.d[j];
            }
            h[[j, m]] = -0.5 * parts.d[j] * e;
            h[[m, j]] = h[[j, m]];
        }
        g[m] = 0.5 * self.n_minus_mp - 0.5 * dp * e;
        h[[m, m]] = 0.5 * dp * e;
        Ok((self.joint_value(v_p, dp, tau), g, h))
    }

    fn axis_bounds(&self) -> Option<Vec<(f64, f64)>> {
        let mut b = self.profiled.axis_bounds()?;
        b.push((f64::NEG_INFINITY, f64::INFINITY));
        Some(b)
    }

    fn stats(&self) -> Option<&crate::stats::FitStats> {
        self.profiled.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::{Additive, DesignStrategy, TermSpec};
    use crate::fit::gaussian::gaussian_score;
    use crate::inner::CholeskySolver;

    /// The joint gradient and Hessian against central differences of the
    /// joint value, away from the profile line (τ ≠ log φ̂) where the
    /// un-profiling terms matter.
    #[test]
    fn joint_derivatives_match_finite_differences() {
        let n = 250;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n - 1) as f64;
            if j == 0 {
                t
            } else {
                (5.1 * t).cos() * 0.5 + 0.5
            }
        });
        let y = Array1::from_shape_fn(n, |i| {
            let t = i as f64 / (n - 1) as f64;
            (6.0 * t).sin()
                + 0.3 * (x[[i, 1]] - 0.5)
                + 0.2 * ((i * 7919 % 113) as f64 / 113.0 - 0.5)
        });
        let prep = Additive {
            terms: vec![
                TermSpec::Cr {
                    col: 0,
                    k: 10,
                    pc: None,
                    by: None,
                },
                TermSpec::Cr {
                    col: 1,
                    k: 8,
                    pc: None,
                    by: None,
                },
            ],
        }
        .prepare(x.view())
        .unwrap();
        let profiled = gaussian_score::<CholeskySolver>(
            &prep.x_design,
            &prep.s_list,
            &prep.rank_s_list,
            prep.mp,
            &prep.log_pseudo_det_s_list,
            y.view(),
            None,
        );
        let joint = JointScaleScore::new(&profiled, n, prep.mp);
        let theta = Array1::from_vec(vec![1.3, 4.2, (0.02_f64).ln()]);
        let (_, g, h) = joint.value_grad_hess(&theta).unwrap();
        let eps = 1e-4;
        for i in 0..theta.len() {
            let mut tp = theta.clone();
            let mut tm = theta.clone();
            tp[i] += eps;
            tm[i] -= eps;
            let fd_g = (joint.value(&tp).unwrap() - joint.value(&tm).unwrap()) / (2.0 * eps);
            assert!(
                (fd_g - g[i]).abs() < 1e-5 * (1.0 + g[i].abs()),
                "g[{i}] = {:.8e}, FD {fd_g:.8e}",
                g[i]
            );
            let (_, gp, _) = joint.value_grad_hess(&tp).unwrap();
            let (_, gm, _) = joint.value_grad_hess(&tm).unwrap();
            for j in 0..theta.len() {
                let fd_h = (gp[j] - gm[j]) / (2.0 * eps);
                assert!(
                    (fd_h - h[[i, j]]).abs() < 1e-5 * (1.0 + h[[i, j]].abs()),
                    "H[{i},{j}] = {:.8e}, FD {fd_h:.8e}",
                    h[[i, j]]
                );
            }
        }
    }
}
