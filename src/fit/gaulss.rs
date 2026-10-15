//! Gaussian location-scale (`gaulss`) — the first GAMLSS (multi-linear-
//! predictor) family.
//!
//! `gaulss` models `y ~ N(μ(x), σ(x)²)` with **two** linear predictors, each
//! with its own smooth terms and smoothing parameters:
//!
//! ```text
//!   η₁ = μ(x)        (identity link)
//!   η₂ = log σ(x)    (log link)
//! ```
//!
//! **Why this needs no dense block-Newton.** The Gaussian location-scale
//! Fisher information is block-diagonal — `E[∂²ℓ/∂η₁∂η₂] = 0` (location and
//! scale are orthogonal parameters for the Gaussian). So the joint penalised
//! MLE decomposes into an *alternation* of two single-predictor penalised
//! weighted-Gaussian REML fits, each of which gamrs already does:
//!
//!   * **block 1 (μ):** Gaussian REML of `y` on `X₁` with prior weights
//!     `1/σ²(x)` — the GLS reweighting is the efficiency gain a two-stage
//!     (unweighted-μ) estimator lacks.
//!   * **block 2 (log σ):** Fisher scoring for the scale, which (constant
//!     expected weight) reduces to a Gaussian REML fit of the working
//!     response `z₂ = η₂ + ½(r²/σ² − 1)` on `X₂`, with `r = y − μ`.
//!
//! Iterated to a joint fixed point. This is the natural "outer iteration"
//! GAMLSS shape and reuses [`fit_with_design`] per block verbatim. (A general
//! dense-block Newton is only required for *non-orthogonal* GAMLSS families,
//! e.g. `shash`; this module is the orthogonal-family case.)
//!
//! Validated against mgcv `gaulss`: μ̂/σ̂ and OOS pinball match to ~3-4 decimals.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::design::{Additive, TermSpec};
use crate::error::{GamrsError, Result};
use crate::family::gaussian_identity;
use crate::fit::{fit_with_design, FittedGam};

/// `E[log|Z|]` for `Z ~ N(0, 1)`. `log σ̂ = E[log|r|] − E[log|Z|]`, so we
/// subtract this (negative) constant from `log|r|` to debias the init.
const E_LOG_ABS_NORMAL: f64 = -0.6351814227307388;

/// Convergence controls for the gaulss outer alternation.
#[derive(Clone, Copy)]
pub struct GaulssOpts {
    pub max_iter: usize,
    /// Max-|Δη₂| tolerance on the log-scale predictor between alternations.
    pub tol: f64,
}

impl Default for GaulssOpts {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-6,
        }
    }
}

/// A fitted Gaussian location-scale model: two reused single-predictor
/// Gaussian REML blocks (`loc` = μ, `scale` = log σ) plus convergence info.
/// Block-orthogonality means each block's `vcov` is its own diagonal block of
/// the joint covariance.
pub struct GaulssFit {
    /// Location block: `predict` yields `η₁ = μ̂(x)` (identity link).
    pub loc: FittedGam,
    /// Scale block: `predict` yields `η₂ = log σ̂(x)`.
    pub scale: FittedGam,
    pub n_iters: usize,
    pub converged: bool,
}

impl GaulssFit {
    /// Conditional mean `μ̂(x)`.
    pub fn predict_loc(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        self.loc.predict(x)
    }

    /// Conditional standard deviation `σ̂(x) = exp(η₂)`.
    pub fn predict_sigma(&self, x: ArrayView2<f64>) -> Result<Array1<f64>> {
        Ok(self.scale.predict(x)?.mapv(f64::exp))
    }

    /// `τ`-quantile `q_τ(x) = μ̂(x) + σ̂(x)·Φ⁻¹(τ)`. Monotone in `τ` with
    /// `σ̂ > 0`, so quantile bands never cross and one fit serves every `τ`.
    pub fn predict_quantile(&self, x: ArrayView2<f64>, tau: f64) -> Result<Array1<f64>> {
        if !(tau > 0.0 && tau < 1.0) {
            return Err(GamrsError::InvalidParameter(format!(
                "gaulss quantile tau must be in (0, 1); got {tau}"
            )));
        }
        let z = norm_ppf(tau);
        let mu = self.predict_loc(x)?;
        let sigma = self.predict_sigma(x)?;
        Ok(&mu + &(&sigma * z))
    }
}

/// Fit a Gaussian location-scale model by orthogonal alternating Fisher
/// scoring. `mu_terms` / `sigma_terms` are the additive smooth specs for the
/// location and scale predictors (each can be multi-smooth).
pub fn fit_gaulss(
    mu_terms: Vec<TermSpec>,
    sigma_terms: Vec<TermSpec>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    opts: GaulssOpts,
) -> Result<GaulssFit> {
    let n = y.len();
    if n == 0 {
        return Err(GamrsError::InvalidParameter(
            "gaulss: empty response".into(),
        ));
    }
    let y_owned = y.to_owned();

    // Response scale for the log-σ init floor (guard log(0) on exact-fit rows).
    let y_mean = y_owned.sum() / n as f64;
    let y_sd = (y_owned.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let floor = 1e-3 * y_sd.max(1e-12);

    // ── Init: unweighted μ fit, then a smooth log-σ start from |residuals|. ──
    let mut loc = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: mu_terms.clone(),
        },
        x,
        y,
        None,
    )?;
    let mu0 = loc.predict(x)?;
    let log_abs_r0: Array1<f64> = (0..n)
        .map(|i| (y_owned[i] - mu0[i]).abs().max(floor).ln() - E_LOG_ABS_NORMAL)
        .collect();
    let mut scale = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: sigma_terms.clone(),
        },
        x,
        log_abs_r0.view(),
        None,
    )?;
    let mut eta2 = scale.predict(x)?;

    // ── Outer alternation. ──
    let mut converged = false;
    let mut iters = 0usize;
    for it in 0..opts.max_iter {
        iters = it + 1;
        let eta2_old = eta2.clone();

        // block 1 (μ): weighted Gaussian REML, prior weights 1/σ².
        let inv_sig2: Array1<f64> = eta2.mapv(|e| (-2.0 * e).exp());
        loc = fit_with_design(
            gaussian_identity(),
            Additive {
                terms: mu_terms.clone(),
            },
            x,
            y,
            Some(inv_sig2.view()),
        )?;
        let mu = loc.predict(x)?;

        // block 2 (log σ): Fisher-scoring IRLS = Gaussian REML of the working
        // response z₂ = η₂ + ½(r²/σ² − 1) on X₂.
        let z2: Array1<f64> = (0..n)
            .map(|i| {
                let r = y_owned[i] - mu[i];
                let s2 = (2.0 * eta2[i]).exp();
                eta2[i] + 0.5 * (r * r / s2 - 1.0)
            })
            .collect();
        scale = fit_with_design(
            gaussian_identity(),
            Additive {
                terms: sigma_terms.clone(),
            },
            x,
            z2.view(),
            None,
        )?;
        eta2 = scale.predict(x)?;

        let dmax = eta2
            .iter()
            .zip(eta2_old.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if dmax < opts.tol {
            converged = true;
            break;
        }
    }

    // `converged` reflects BOTH the outer alternation AND the two block fits —
    // an alternation that hit tolerance is still suspect if a block's inner
    // REML Newton bailed (mirrors `quantile.rs`'s `outer && final_fit`).
    let converged = converged && loc.converged && scale.converged;
    Ok(GaulssFit {
        loc,
        scale,
        n_iters: iters,
        converged,
    })
}

/// Standard-normal inverse CDF via Acklam's rational approximation
/// (abs error < 1.15e-9 across (0, 1)) — dependency-free `Φ⁻¹` for quantiles.
// Acklam's coefficients are mirrored verbatim from the reference; keep the
// published digits (a newer clippy flags the last digit as excessive — it is
// rounded to the nearest f64 either way, so this is provenance, not precision).
#[allow(clippy::excessive_precision)]
fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Axis};

    // Golden-ratio low-discrepancy fractional parts: well-spread, decorrelated
    // covariates so the CR design is well-conditioned (a plain hash clustered
    // and made X'WX singular).
    fn frac(v: f64) -> f64 {
        v - v.floor()
    }
    fn pnormal(u1: f64, u2: f64) -> f64 {
        (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn gaulss_recovers_heteroskedastic_scale_and_covers() {
        // y = sin(2πx0) + 2(x1−½) + (0.1 + 0.4 x0)·N(0,1): heteroskedastic in x0.
        let n = 800;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        let mut true_sigma = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = frac((i as f64 + 0.5) * 0.6180339887498949);
            let x1 = frac((i as f64 + 0.5) * 0.3819660112501051 + 0.137);
            x[[i, 0]] = x0;
            x[[i, 1]] = x1;
            let mu = (2.0 * std::f64::consts::PI * x0).sin() + 2.0 * (x1 - 0.5);
            let sg = 0.1 + 0.4 * x0;
            true_sigma[i] = sg;
            // Decorrelated uniforms for Box-Muller from two more golden sequences.
            let u1 = frac((i as f64 + 0.5) * 0.7548776662466927 + 0.31);
            let u2 = frac((i as f64 + 0.5) * 0.5698402909980532 + 0.59);
            y[i] = mu + sg * pnormal(u1, u2);
        }
        let terms = vec![
            TermSpec::Cr {
                col: 0,
                k: 10,
                pc: None,
                by: None,
            },
            TermSpec::Cr {
                col: 1,
                k: 10,
                pc: None,
                by: None,
            },
        ];
        let sterms = vec![
            TermSpec::Cr {
                col: 0,
                k: 6,
                pc: None,
                by: None,
            },
            TermSpec::Cr {
                col: 1,
                k: 6,
                pc: None,
                by: None,
            },
        ];
        let fit = fit_gaulss(terms, sterms, x.view(), y.view(), GaulssOpts::default())
            .expect("gaulss fit");
        assert!(
            fit.converged,
            "gaulss did not converge (iters={})",
            fit.n_iters
        );

        // σ̂(x) should track the true heteroskedastic scale (corr > 0.8).
        let sig = fit.predict_sigma(x.view()).unwrap();
        let sm = sig.mean().unwrap();
        let tm = true_sigma.mean().unwrap();
        let (mut cov, mut vs, mut vt) = (0.0, 0.0, 0.0);
        for i in 0..n {
            cov += (sig[i] - sm) * (true_sigma[i] - tm);
            vs += (sig[i] - sm).powi(2);
            vt += (true_sigma[i] - tm).powi(2);
        }
        let corr = cov / (vs.sqrt() * vt.sqrt());
        assert!(corr > 0.8, "σ̂ vs true σ correlation {corr:.3} too low");

        // Quantile coverage ≈ τ, and no crossing across τ.
        let q10 = fit.predict_quantile(x.view(), 0.1).unwrap();
        let q50 = fit.predict_quantile(x.view(), 0.5).unwrap();
        let q90 = fit.predict_quantile(x.view(), 0.9).unwrap();
        for i in 0..n {
            assert!(
                q10[i] <= q50[i] && q50[i] <= q90[i],
                "quantiles crossed at i={i}"
            );
        }
        let cov10 = (0..n).filter(|&i| y[i] <= q10[i]).count() as f64 / n as f64;
        let cov90 = (0..n).filter(|&i| y[i] <= q90[i]).count() as f64 / n as f64;
        assert!((cov10 - 0.1).abs() < 0.06, "τ=0.1 coverage {cov10:.3}");
        assert!((cov90 - 0.9).abs() < 0.06, "τ=0.9 coverage {cov90:.3}");
        let _ = x.index_axis(Axis(0), 0); // silence unused Axis import on some toolchains
    }
}
