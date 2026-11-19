//! A gaussian REML criterion with two local optima (scripts/r/gen_reml_start_basin_fixture.R).
//! The search started at λ = 1 ended in the worse one: the heavy-tailed lot-size
//! smooth kept edf 3.47 and REML was 2.4 units above mgcv's. Started where mgcv
//! starts (`initial.sp`), it lands on mgcv's optimum, the lot smooth a straight line.

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
    k: Vec<usize>,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
}

#[derive(Deserialize)]
struct MgcvOutput {
    fitted: Vec<f64>,
    reml: f64,
    edf_per_term: Vec<f64>,
}

#[test]
fn gaussian_reml_lands_in_mgcvs_basin() {
    let f: Fixture = serde_json::from_str(include_str!(
        "fixtures/reml_start_basin_gaussian_n1500.json"
    ))
    .unwrap();
    let n = f.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, f.inputs.k.len()), f.inputs.x_train.concat()).unwrap();
    let y = Array1::from(f.inputs.y_train.clone());
    let terms = f
        .inputs
        .k
        .iter()
        .enumerate()
        .map(|(col, &k)| TermSpec::Cr {
            col,
            k,
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
    .unwrap();
    let m = &f.mgcv_output;

    assert!(
        (fit.reml_value - m.reml).abs() < 1e-2,
        "REML {} vs mgcv {}",
        fit.reml_value,
        m.reml
    );
    for (j, (got, want)) in fit.edf_per_term.iter().zip(&m.edf_per_term).enumerate() {
        assert!(
            (got - want).abs() < 0.05,
            "term {j}: edf {got} vs mgcv {want}"
        );
    }
    // `lot`: a straight line in mgcv's fit, edf 3.47 from λ = 1.
    assert!(
        fit.edf_per_term[5] < 1.05,
        "lot edf {}",
        fit.edf_per_term[5]
    );
    let fitted = fit.predict(x.view()).unwrap();
    let sd = {
        let mean = y.mean().unwrap();
        (y.mapv(|v| (v - mean).powi(2)).sum() / (n as f64 - 1.0)).sqrt()
    };
    let worst = fitted
        .iter()
        .zip(&m.fitted)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(worst < 5e-3 * sd, "fitted {worst} from mgcv (sd {sd})");
}
