//! Parity test for gamrs's multi-smooth `Additive` path with the NegBin
//! (profile-θ) family against an mgcv `nb()` 2-D additive fixture
//! (`y ~ s(x0) + s(x1)`). Confirms the shape-aware REML score handles a
//! per-term penalty list AND a profiled shape parameter simultaneously.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::negbin_log;
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

#[test]
fn additive_2d_nb_n600_k8_cr() {
    let fx = load_fixture("2d_nb_log_n600_k8_cr");
    let n = fx.inputs.x_train.len();
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * 2);
    for row in &fx.inputs.x_train {
        x_flat.push(row[0]);
        x_flat.push(row[1]);
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());

    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: fx.inputs.k[0],
            pc: None,
        },
        TermSpec::Cr {
            col: 1,
            k: fx.inputs.k[1],
            pc: None,
        },
    ];
    // init θ = 5.0 (mirrors parity_negbin.rs); θ is profiled jointly.
    let fit = fit_with_design(
        negbin_log(5.0),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive NegBin fit failed");
    assert!(fit.converged, "outer Newton did not converge");
    assert_eq!(fit.rho.len(), 2, "one smoothing param per term");
    assert_eq!(fit.edf_per_term.len(), 2);

    // gamrs predict returns η (log link); mgcv predictions_train is μ.
    let eta = fit.predict(x.view()).expect("predict failed");
    let mu: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();
    let rel = max_rel_err(&mu, &fx.mgcv_output.predictions_train);
    println!(
        "[additive nb n600 k8] max_rel = {rel:.3e}; ρ̂ = [{:.3}, {:.3}]; edf = {:.2}; iters = {}",
        fit.rho[0], fit.rho[1], fit.edf_total, fit.n_iters,
    );
    // Observed 4.3e-6 (1.4e-3 before 0.14.6). Multi-smooth NB + profiled θ.
    assert!(rel < 5e-5, "additive NB μ rel error {rel:.3e} exceeds 5e-5");
}
