//! Parity test for gamrs's multi-smooth `Additive` path with the `scat`
//! (scaled-t) family against mgcv `scat()` 2-D / 3-D additive fixtures
//! (`y ~ s(x0) + s(x1) [+ s(x2)]`, identity link, heavy-tailed t noise).
//!
//! Closes the README's "scat / TDist multi-smooth … reference parity tests
//! are pending" gap. scat is the hardest convergence regime in the battery
//! (joint Newton over `[log λ…, log σ², log(ν−min.df)]`), so the bound is
//! relaxed relative to the GLM families — but it is a *measured*, locked
//! bound, not a smoke test. Both bounds tightened ~16× and ~8× when
//! `TDist::score_rank_adjustment`'s −1 came out (see `parity_scat.rs`).

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::tdist_identity;
use gamrs::fit_with_design;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    mgcv_output: MgcvOutput,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
}

#[derive(Deserialize)]
struct MgcvOutput {
    predictions_train: Vec<f64>,
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn max_rel_err(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

fn run_scat_additive(name: &str, d: usize, bound: f64) {
    let fx = load_fixture(name);
    let n = fx.inputs.x_train.len();
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * d);
    for row in &fx.inputs.x_train {
        x_flat.extend_from_slice(&row[..d]);
    }
    let x = Array2::from_shape_vec((n, d), x_flat).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let y_var = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n as f64);

    let terms: Vec<TermSpec> = (0..d)
        .map(|j| TermSpec::Cr {
            col: j,
            k: fx.inputs.k[j],
            pc: None,
            by: None,
        })
        .collect();
    // Seed ν = 5 (near scat's typical optimum), σ² = var(y)/10 — the joint
    // Newton profiles both. Mirrors the 1-D scat parity seeding.
    let fit = fit_with_design(
        tdist_identity(5.0, y_var * 0.1),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive scat fit failed");
    assert_eq!(fit.rho.len(), d, "one smoothing param per term");

    // Identity link — gamrs predict returns μ directly (mgcv predictions are μ).
    let mu = fit.predict(x.view()).expect("predict failed");
    let rel = max_rel_err(mu.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    println!(
        "[additive scat {name}] max_rel = {rel:.3e}; σ̂² = {:.4}; edf = {:.2}; \
         iters = {}; converged = {}",
        fit.scale, fit.edf_total, fit.n_iters, fit.converged,
    );
    assert!(
        rel < bound,
        "additive scat μ rel error {rel:.3e} exceeds {bound:.1e}"
    );
}

#[test]
fn additive_2d_scat_n600_k8_cr() {
    // Bar 5e-5: observed 5.502e-6 (σ̂²=0.2496 vs mgcv σ²=0.25, edf 8.98 vs
    // mgcv's 8.9798, converged). ~9x headroom — loose enough for last-digit
    // drift across a refactor, tight enough that this fixture no longer
    // passes through a real regression.
    //
    // This used to carry a 1.5e-3 bar justified by a "~1e-3 residual in
    // scat's ρ-gradient at moderate ρ". That residual is gone: the same
    // probe (`score_tests.rs::tdist_analytic_rho_grad_matches_fd`) now reads
    // analytic +1.557897e-1 vs FD +1.557897e-1 at ρ=0, abs err 5.6e-8. The
    // observed-curvature criterion closed it, and the fixture improved 196x.
    run_scat_additive("2d_scat_identity_n600_k8_cr", 2, 5e-5);
}

#[test]
fn additive_3d_scat_n800_k8_cr() {
    // Bar 1e-3: observed 8.281e-5 (σ̂²=0.1615 vs mgcv σ²=0.162, converged).
    // Left as-is deliberately — ~12x headroom is already the band we want,
    // and this fixture is the loosest of the scat set for a reason: 3 margins
    // means 3 flat-ish ρ axes, so it drifts more than the 2-D one under any
    // change to the outer loop.
    run_scat_additive("3d_scat_identity_n800_k8_cr", 3, 1e-3);
}
