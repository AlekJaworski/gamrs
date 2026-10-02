//! Phase 6 parity test: gamrs `fit_negbin_cr` vs mgcv `nb()` on 1-D NegBin
//! + log. mgcv estimates θ via profile-likelihood; gamrs does joint Newton
//! on `[log λ, log θ]`. Compares μ predictions.
#![allow(clippy::doc_lazy_continuation)]

use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::path::PathBuf;

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

fn load(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn max_rel_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (y.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn nb_log_n300_k10_cr() {
    let fx = load("1d_nb_log_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];
    let fit = gamrs::fit(
        gamrs::family::negbin_log(/*init_theta=*/ 5.0),
        x.view(),
        y.view(),
        None,
        k,
    )
    .unwrap();
    assert!(fit.converged, "NegBin outer did not converge");
    let eta = fit.predict(x.view()).unwrap();
    let mu_gamrs: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();
    let rel = max_rel_err(&mu_gamrs, &fx.mgcv_output.predictions_train);
    let abs_e = max_abs_err(&mu_gamrs, &fx.mgcv_output.predictions_train);
    println!(
        "[nb n300 k10] max_rel = {rel:.3e}; max_abs = {abs_e:.3e}; θ̂ gamrs = {:.3}; ρ̂ = {:.3}; iters = {}; edf = {:.2}",
        fit.scale, fit.rho[0], fit.n_iters, fit.edf_total,
    );
    // Observed 2.5e-7. The bar was 1e-2 until 0.14.6, wide enough to hide a
    // wrong criterion: the θ probes scored a frozen β, and log|H| dropped its
    // β-chain.
    assert!(rel < 1e-5, "NegBin μ rel error {rel:.3e} exceeds 1e-5");
}
