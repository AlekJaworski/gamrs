//! Parity tests for gamrs's multi-smooth `Additive` Tweedie path against
//! mgcv 2-D additive fixtures (`y ~ s(x0) + s(x1)`), in BOTH modes:
//!
//! - profile-p (mgcv `tw()`): p estimated jointly with φ and the smoothing
//!   params — `tweedie_log(...)`, 2 shape params `[log φ, p_transform]`.
//! - fixed-p (mgcv `Tweedie(p=1.5)`): p held constant, only φ + λ estimated —
//!   `tweedie_log_fixed_p(...)`, 1 shape param `[log φ]`.
//!
//! The multi-smooth Tweedie bar (~1.5e-2) is looser than single-smooth
//! (`parity_tweedie.rs`, 5e-3): the additive penalty list adds outer-Newton
//! dof, so the converged (ρ, φ, p) lands marginally off mgcv's. The same
//! looseness appears whether p is profiled or fixed, confirming it is
//! additive-penalty drift, not the p path.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::{tweedie_log, tweedie_log_fixed_p};
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

fn build_xy(fx: &Fixture) -> (Array2<f64>, Array1<f64>) {
    let n = fx.inputs.x_train.len();
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * 2);
    for row in &fx.inputs.x_train {
        x_flat.push(row[0]);
        x_flat.push(row[1]);
    }
    (
        Array2::from_shape_vec((n, 2), x_flat).unwrap(),
        Array1::from_vec(fx.inputs.y_train.clone()),
    )
}

#[test]
fn additive_2d_tweedie_profile_p_n600_k8_cr() {
    let fx = load_fixture("2d_tw_profile_log_n600_k8_cr");
    let (x, y) = build_xy(&fx);
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: fx.inputs.k[0],
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: fx.inputs.k[1],
            pc: None,
            by: None,
        },
    ];
    // profile-p: p estimated jointly.
    let fit = fit_with_design(
        tweedie_log(1.5, 1.0),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive Tweedie (profile-p) fit failed");
    assert_eq!(fit.rho.len(), 2);
    let eta = fit.predict(x.view()).expect("predict failed");
    let mu: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();
    let rel = max_rel_err(&mu, &fx.mgcv_output.predictions_train);
    println!(
        "[additive tw profile-p n600 k8] max_rel = {rel:.3e}; ρ̂ = [{:.3}, {:.3}]; edf = {:.2}",
        fit.rho[0], fit.rho[1], fit.edf_total,
    );
    // Bar 1.5e-2: observed ~1.2e-2.
    assert!(
        rel < 1.5e-2,
        "additive Tweedie profile-p μ rel {rel:.3e} exceeds 1.5e-2"
    );
}

#[test]
fn additive_2d_tweedie_fixed_p_n600_k8_cr() {
    let fx = load_fixture("2d_tw_fixed_p15_log_n600_k8_cr");
    let (x, y) = build_xy(&fx);
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: fx.inputs.k[0],
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: fx.inputs.k[1],
            pc: None,
            by: None,
        },
    ];
    // fixed-p: p held at 1.5, only φ + λ estimated.
    let fit = fit_with_design(
        tweedie_log_fixed_p(1.5, 1.0),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive Tweedie (fixed-p) fit failed");
    assert_eq!(fit.rho.len(), 2);
    let eta = fit.predict(x.view()).expect("predict failed");
    let mu: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();
    let rel = max_rel_err(&mu, &fx.mgcv_output.predictions_train);
    println!(
        "[additive tw fixed-p n600 k8] max_rel = {rel:.3e}; ρ̂ = [{:.3}, {:.3}]; edf = {:.2}",
        fit.rho[0], fit.rho[1], fit.edf_total,
    );
    // Bar 1.5e-2: observed ~1.2e-2 (same looseness as profile-p).
    assert!(
        rel < 1.5e-2,
        "additive Tweedie fixed-p μ rel {rel:.3e} exceeds 1.5e-2"
    );
}
