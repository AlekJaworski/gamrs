//! Parity test for gamrs's `Additive` multi-smooth path against an
//! mgcv-generated 2-D additive Gaussian fixture (`y ~ s(x0) + s(x1)`).
//!
//! Same shape as `parity_gaussian.rs` but the fit goes through
//! `fit_with_design(gaussian_identity(), Additive { terms: [Cr {col:0, k},
//! Cr {col:1, k}] }, …)` instead of the default single-smooth `Cr`. The
//! per-term EDF sums to `edf_total - 1` (the intercept's fixed dof) so we
//! also exercise the new `edf_per_term` field.

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

#[test]
fn additive_2d_gaussian_n500_k10_cr() {
    let fx = load_fixture("2d_gaussian_additive_n500_k10_cr");

    // x is `(n, 2)` — column 0 is x0, column 1 is x1.
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
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: fx.inputs.k[1],
            pc: None,
            by: None,
        },
    ];
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive Gaussian fit failed");
    assert!(fit.converged, "outer Newton did not converge");
    assert_eq!(fit.rho.len(), 2, "rho should have one entry per term");
    assert_eq!(fit.lambda.len(), 2);
    assert_eq!(fit.edf_per_term.len(), 2);

    // Per-term EDF sums to ~edf_total - 1 (the intercept's fixed dof of 1
    // sits outside the per-term split). Tolerance absorbs tiny null-space
    // accounting differences.
    let edf_sum: f64 = fit.edf_per_term.iter().sum();
    assert!(
        (edf_sum - (fit.edf_total - 1.0)).abs() < 1e-6,
        "edf_per_term sums to {} but edf_total - 1 = {}",
        edf_sum,
        fit.edf_total - 1.0,
    );

    // Predictions vs mgcv. Multi-smooth fit precision is slightly looser
    // than single-smooth (more outer-Newton dof, more chances for tiny
    // numerical drift) — set bound at 5e-4 (single-smooth bound is 1e-5).
    let pred = fit
        .predict(x.view())
        .expect("Additive predict on training x failed");
    let rel = max_rel_err(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    let scale_rel = (fit.scale - fx.mgcv_output.scale).abs() / fx.mgcv_output.scale.max(1e-12);
    println!(
        "[additive 2d gaussian n500 k10 cr] max_rel = {rel:.3e}; scale_rel = {scale_rel:.3e} \
         (gamrs {:.6e} vs mgcv {:.6e}); ρ̂ = [{:.3}, {:.3}]; edf = ({:.2}, {:.2}); iters = {}",
        fit.scale,
        fx.mgcv_output.scale,
        fit.rho[0],
        fit.rho[1],
        fit.edf_per_term[0],
        fit.edf_per_term[1],
        fit.n_iters,
    );
    assert!(
        rel < 5e-4,
        "additive μ rel error {rel:.3e} exceeds 5e-4 — possible regression"
    );
    assert!(
        scale_rel < 1e-2,
        "additive φ̂ rel error {scale_rel:.3e} exceeds 1e-2"
    );
}
