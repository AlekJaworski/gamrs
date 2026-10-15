//! High-dimensional (3-D … 10-D) additive Gaussian parity tests against
//! mgcv-generated fixtures (`y ~ s(x0) + s(x1) + … + s(x_{d-1})`, all CR
//! splines, identity link, REML).
//!
//! Mirrors `parity_additive.rs`: build `Additive { terms: [Cr{col,k}, …] }`
//! for every column, fit `gaussian_identity()`, assert convergence, and
//! compare `fit.predict(x)` to `mgcv_output.predictions_train` with the
//! additive `max_rel_err` bar (denominator `|target| + 1`).

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::gaussian_identity;
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
    d: usize,
    k: Vec<usize>,
}

#[derive(Deserialize)]
struct MgcvOutput {
    predictions_train: Vec<f64>,
    scale: f64,
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

/// Load a high-dim additive fixture, fit through the `Additive` CR path, and
/// assert μ-parity against mgcv at `mu_bar`. Returns the observed max-rel error.
fn run_highdim_additive(name: &str, mu_bar: f64) {
    let fx = load_fixture(name);
    let d = fx.inputs.d;
    assert_eq!(fx.inputs.k.len(), d, "fixture k length must equal d");

    let n = fx.inputs.x_train.len();
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * d);
    for row in &fx.inputs.x_train {
        assert_eq!(row.len(), d, "x_train row width must equal d");
        x_flat.extend_from_slice(row);
    }
    let x = Array2::from_shape_vec((n, d), x_flat).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());

    let terms: Vec<TermSpec> = (0..d)
        .map(|c| TermSpec::Cr {
            col: c,
            k: fx.inputs.k[c],
            pc: None,
            by: None,
        })
        .collect();

    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive Gaussian fit failed");

    assert!(fit.converged, "[{name}] outer Newton did not converge");
    assert_eq!(
        fit.rho.len(),
        d,
        "[{name}] rho should have one entry per term"
    );
    assert_eq!(
        fit.lambda.len(),
        d,
        "[{name}] lambda should have one entry per term"
    );
    assert_eq!(
        fit.edf_per_term.len(),
        d,
        "[{name}] edf_per_term length mismatch"
    );

    let pred = fit
        .predict(x.view())
        .expect("Additive predict on training x failed");
    let rel = max_rel_err(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    let scale_rel = (fit.scale - fx.mgcv_output.scale).abs() / fx.mgcv_output.scale.max(1e-12);
    println!(
        "[{name}] d={d} n={n} max_rel = {rel:.3e} (bar {mu_bar:.0e}); \
         scale_rel = {scale_rel:.3e} (gamrs {:.6e} vs mgcv {:.6e}); iters = {}",
        fit.scale, fx.mgcv_output.scale, fit.n_iters,
    );

    assert!(
        rel < mu_bar,
        "[{name}] additive μ rel error {rel:.3e} exceeds {mu_bar:.0e} — possible high-dim regression"
    );
    assert!(
        scale_rel < 1e-2,
        "[{name}] additive φ̂ rel error {scale_rel:.3e} exceeds 1e-2"
    );
}

#[test]
fn additive_3d_gaussian_n800_k10_cr() {
    run_highdim_additive("3d_gaussian_mixed_n800_k10_cr", 5e-4);
}

#[test]
fn additive_4d_gaussian_n1000_k10_cr() {
    run_highdim_additive("4d_gaussian_mixed_n1000_k10_cr", 5e-4);
}

#[test]
fn additive_5d_gaussian_n1500_k8_cr() {
    run_highdim_additive("5d_gaussian_mixed_n1500_k8_cr", 5e-4);
}

#[test]
fn additive_7d_neighbourhoods_compact_n3000() {
    run_highdim_additive("7d_neighbourhoods_compact_n3000", 5e-4);
}

#[test]
fn additive_8d_neighbourhoods_like_n15000() {
    run_highdim_additive("8d_neighbourhoods_like_n15000", 5e-4);
}

#[test]
fn additive_10d_gaussian_n3000_k8_cr() {
    run_highdim_additive("10d_gaussian_n3000_k8_cr", 5e-4);
}
