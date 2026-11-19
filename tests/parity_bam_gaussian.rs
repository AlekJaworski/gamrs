//! `fit_bam` against mgcv `bam(method = "fREML")` on 3-D additive gaussian
//! fixtures (`scripts/r/gen_bam_freml_fixtures.R`). x2 is linear, so its λ sits
//! on a flat REML ridge where only the same Newton path lands on bam's value.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit_bam;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    bam: Reference,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
    weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    sp: Vec<f64>,
    scale: f64,
    predictions_train: Vec<f64>,
    edf_per_term: Vec<f64>,
    edf_total: f64,
    reml: f64,
    iter: usize,
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn check(name: &str) {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(format!("tests/fixtures/{name}.json"));
    let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 3), fx.inputs.x_train.concat()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train);
    let w = fx.inputs.weights.map(Array1::from_vec);
    let terms = (0..3)
        .map(|col| TermSpec::Cr {
            col,
            k: fx.inputs.k[col],
            pc: None,
            by: None,
        })
        .collect();
    let fit = fit_bam(
        gaussian_identity(),
        &Additive { terms },
        x.view(),
        y.view(),
        w.as_ref().map(|w| w.view()),
    )
    .unwrap();
    let r = fx.bam;

    let pred = fit.predict(x.view()).unwrap();
    let fitted = pred
        .iter()
        .zip(&r.predictions_train)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let sp = fit
        .lambda
        .iter()
        .zip(&r.sp)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let edf_term = fit
        .edf_per_term
        .iter()
        .zip(&r.edf_per_term)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let report = format!(
        "{name}: fitted {fitted:.2e} sp {sp:.2e} edf/term {edf_term:.2e} edf {:.2e} \
         reml {:.2e} scale {:.2e} iter {} vs {}",
        rel(fit.edf_total, r.edf_total),
        rel(fit.reml_value, r.reml),
        rel(fit.scale, r.scale),
        fit.n_iters,
        r.iter
    );
    eprintln!("{report}");
    // Measured ≤ 2e-11 on every quantity; the flat-ridge λ gets the most room.
    assert!(fitted < 1e-9, "{report}");
    assert!(sp < 1e-7, "{report}");
    assert!(edf_term < 1e-9, "{report}");
    assert!(rel(fit.edf_total, r.edf_total) < 1e-9, "{report}");
    assert!(rel(fit.reml_value, r.reml) < 1e-10, "{report}");
    assert!(rel(fit.scale, r.scale) < 1e-9, "{report}");
    assert_eq!(fit.n_iters, r.iter, "{report}");
}

#[test]
fn bam_gaussian_flat_ridge_term() {
    check("3d_gaussian_bam_freml_n2000_k10_cr");
}

#[test]
fn bam_gaussian_prior_weights() {
    check("3d_gaussian_weighted_bam_freml_n2000_k10_cr");
}
