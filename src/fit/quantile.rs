//! Quantile/ELF fit helper — qgam warm start + ArmijoElfInner.
//!
//! `fit_quantile_from_prep` is the shared ELF driver consumed by the
//! `FamilyFitWithSolver` impl for `ElfLoss` in `canonical.rs`. Not a
//! public entry point — the canonical surface is `gamrs::fit(...)`.

use std::marker::PhantomData;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::design::PreparedDesign;
use crate::error::{GamrsError, Result};
use crate::family::{elf_identity, ElfLoss};
use crate::inner::{ArmijoElfInner, ArmijoElfOpts, GaussianInnerFit, LinearSolver};
use crate::outer::NewtonWithHalving;
use crate::score::{EnvelopeScore, FixedAtOneProfile};
use crate::traits::{InnerSolver, OuterSolver};

use super::{compute_edf, compute_edf_per_term, compute_vcov, FittedGam, LinkKind};

/// qgam-style warm start (v0.x `fit_pirls_quantile_impl` Steps 1-4):
/// Gaussian-init β + empirical τ-quantile shift. Returns `(β_init,
/// σ̂_residuals²)` — `σ̂²` feeds the (σ, λ_elf) heuristic in
/// `derive_elf_sigma_lambda`.
fn qgam_warm_start(
    prep: &PreparedDesign,
    y: ArrayView1<f64>,
    tau: f64,
) -> Result<(Array1<f64>, f64)> {
    use crate::inner::{chol_back_solve, chol_forward_solve};
    use ndarray_linalg::{Cholesky, UPLO};

    let n = y.len();
    let p = prep.x_design.ncols();

    // 1) Solve unpenalised-loss Gaussian at λ_pen = 1 (default outer start).
    let xtx: Array2<f64> = prep.x_design.t().dot(&prep.x_design);
    // S_total at ρ = 0 (i.e. λ_j = 1 for every term). `combined_s` with a
    // zero rho of length `s_list.len()` yields `Σ_j S_j` (each `exp(0)=1`),
    // so this warm start is correct for both single- and multi-smooth.
    let rho_init: Array1<f64> = Array1::zeros(prep.s_list.len());
    let s_total_init = crate::design::combined_s(&prep.s_list, &rho_init, prep.x_design.ncols());
    let mut a_gauss = &xtx + &s_total_init;
    let mut max_diag = 1.0_f64;
    for i in 0..p {
        max_diag = max_diag.max(a_gauss[[i, i]].abs());
    }
    let ridge_g = 1e-7 * max_diag;
    for i in 0..p {
        a_gauss[[i, i]] += ridge_g;
    }
    let xty: Array1<f64> = prep.x_design.t().dot(&y.to_owned());
    let chol_g = a_gauss
        .cholesky(UPLO::Lower)
        .map_err(|e| GamrsError::SingularSystem(format!("ELF warm-start Cholesky: {e}")))?;
    let zg = chol_forward_solve(&chol_g, xty.view());
    let beta_gauss = chol_back_solve(&chol_g, zg.view());

    // 2) Residuals + variance.
    let mu_gauss: Array1<f64> = prep.x_design.dot(&beta_gauss);
    let r_vec: Vec<f64> = y
        .iter()
        .zip(mu_gauss.iter())
        .map(|(&yi, &mi)| yi - mi)
        .collect();
    let sigma2_hat = r_vec.iter().map(|&ri| ri * ri).sum::<f64>() / (n as f64).max(1.0);

    // 3) Empirical τ-quantile of residuals → per-obs constant shift on η.
    let mut r_sorted = r_vec.clone();
    r_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let qi_r = ((n as f64 - 1.0) * tau).round() as usize;
    let q_r = r_sorted[qi_r.min(n.max(1) - 1)];

    // 4) β_init = β_gauss + (X'X + λS + ridge)⁻¹ · X'·(q_r · 1)
    let q_const = Array1::from_elem(n, q_r);
    let xtq = prep.x_design.t().dot(&q_const);
    let zq = chol_forward_solve(&chol_g, xtq.view());
    let delta_beta = chol_back_solve(&chol_g, zq.view());
    let beta_init: Array1<f64> = &beta_gauss + &delta_beta;

    Ok((beta_init, sigma2_hat))
}

/// Pick `(σ, λ_elf)` for the ELF loss. Uses qgam's `co` formula with
/// `err=0.05` and a τ-tail widening factor when the caller passes
/// non-positive `init_sigma` / `init_lambda`.
fn derive_elf_sigma_lambda(
    init_sigma: f64,
    init_lambda: f64,
    sigma2_hat: f64,
    tau: f64,
) -> (f64, f64) {
    if init_sigma > 0.0 {
        let lambda_eff = if init_lambda > 0.0 {
            init_lambda
        } else {
            init_sigma
        };
        return (init_sigma, lambda_eff);
    }
    let err = 0.05_f64;
    let sigma2_floor = sigma2_hat.max(1e-6);
    let co_default =
        err * (2.0 * std::f64::consts::PI * sigma2_floor).sqrt() / (2.0 * 2.0_f64.ln());
    let tail_scale = (1.0 / (4.0 * tau * (1.0 - tau))).max(1.0);
    let sigma_auto = co_default * tail_scale;
    let lambda_auto = if init_lambda > 0.0 {
        init_lambda
    } else {
        sigma_auto
    };
    (sigma_auto, lambda_auto)
}

/// Population standard deviation of `y`, or 1 when it is zero or not finite.
fn response_scale(y: ArrayView1<f64>) -> f64 {
    let n = y.len().max(1) as f64;
    let mean = y.sum() / n;
    let sd = (y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    if sd.is_finite() && sd > 0.0 {
        sd
    } else {
        1.0
    }
}

/// ELF (quantile) driver. `tau`, `init_sigma`, `init_lambda` are read
/// off `family.loss` at the canonical dispatch site.
pub(crate) fn fit_quantile_from_prep<S: LinearSolver>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    tau: f64,
    init_sigma: f64,
    init_lambda: f64,
) -> Result<FittedGam> {
    if !(tau > 0.0 && tau < 1.0) {
        return Err(GamrsError::InvalidParameter(format!(
            "Quantile (ELF) tau must be in the open interval (0, 1) — \
             0/1 are degenerate (point mass at min/max); got tau={tau}"
        )));
    }

    let n = x.nrows();
    let n_terms = prep.s_list.len();

    // Fit in units of sd(y). The ELF working weights scale as 1/y², so in raw
    // units the optimal λ moves by 1/c² when y is multiplied by c, while the
    // outer Newton starts at ρ = 0 inside a fixed box: on house prices in
    // dollars it stopped at λ ≈ 1 with the smooth shrunk to nothing. A
    // quantile is equivariant to rescaling y, so the fit is done on y/s and
    // mapped back.
    let y_scale = response_scale(y);
    let y_std: Array1<f64> = y.mapv(|v| v / y_scale);
    let y = y_std.view();
    let init_sigma = init_sigma / y_scale;
    let init_lambda = init_lambda / y_scale;

    let (beta_init, sigma2_hat) = qgam_warm_start(&prep, y, tau)?;
    let (sigma, lambda_elf) = derive_elf_sigma_lambda(init_sigma, init_lambda, sigma2_hat, tau);

    // ─── Outer Newton on ρ = log λ_pen ───
    let prior = prior_weights.map(|w| w.to_owned());
    let inner = ArmijoElfInner::<S> {
        x_design: prep.x_design.clone(),
        y: y.to_owned(),
        prior_weights: prior,
        s_list: prep.s_list.clone(),
        family: elf_identity(tau, sigma, lambda_elf),
        opts: ArmijoElfOpts::default(),
        beta_init: Some(beta_init),
        _solver: PhantomData,
    };
    let score = EnvelopeScore::<ElfLoss, ArmijoElfInner<S>, FixedAtOneProfile, S>::with_inner(
        inner,
        ElfLoss {
            tau,
            sigma,
            lambda: lambda_elf,
        },
        FixedAtOneProfile,
        y.to_owned(),
        prep.s_list.clone(),
        prep.rank_s_list.clone(),
        prep.mp,
        prep.log_pseudo_det_s_list.clone(),
    );

    crate::outer::reject_unsupported_algorithm("quantile (ELF)")?;
    let outer_solver =
        NewtonWithHalving::new(crate::outer::resolve_tuning(&score.loss).to_newton_opts());
    let outer = outer_solver.minimize(&score, Array1::<f64>::zeros(n_terms))?;

    let final_fit: GaussianInnerFit<S> = score.inner.fit(&outer.theta)?;
    let edf = compute_edf(&prep.x_design, &final_fit.a_weights, &final_fit);
    let edf_per_term = compute_edf_per_term(
        &prep.s_list,
        &outer.theta,
        prep.x_design.ncols(),
        &final_fit,
    );

    // scale = σ (the ELF likelihood scale). Quantile users want it for
    // diagnostics; mgcv reports `scale = 1`, we deliberately diverge.
    // Back to the units of y: β and σ scale by s; the working weights carry
    // 1/s², so the raw-unit λ is λ/s² and `A⁻¹·σ` picks up s³.
    let vcov = compute_vcov(&final_fit, sigma) * y_scale.powi(3);
    let rho_vec: Array1<f64> = outer.theta.mapv(|r| r - 2.0 * y_scale.ln());
    let lambda_vec: Array1<f64> = rho_vec.iter().map(|&r| r.exp()).collect();
    Ok(FittedGam {
        beta: final_fit.beta * y_scale,
        rho: rho_vec,
        lambda: lambda_vec,
        scale: sigma * y_scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged && final_fit.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind: LinkKind::Identity,
        shape_params: Array1::zeros(0),
        stats: score.stats.snapshot(),
    })
}
