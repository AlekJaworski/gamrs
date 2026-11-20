//! Negative binomial and Tweedie REML fits whose criterion has two local optima
//! (scripts/r/gen_reml_start_extended_fixtures.R). Started where mgcv starts
//! (`initial.spg`: `initial.sp` on √w·X, w at the family's `mustart` and starting
//! θ or p), they land on mgcv's optimum. From the previous start the lot-like nb
//! fit stopped 228 REML units above it, its top lot size fitted at 1.6e5 against
//! mgcv's 5,048; the other two landed on the other optimum, edf off by up to 2.0.

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::{negbin_log, tweedie_log};
use gamrs::{fit_with_design, FittedGam};

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

fn load(json: &str) -> (Fixture, Array2<f64>, Array1<f64>, Additive) {
    let f: Fixture = serde_json::from_str(json).unwrap();
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
    (f, x, y, Additive { terms })
}

fn check(fit: &FittedGam, x: &Array2<f64>, m: &MgcvOutput) {
    assert!(fit.converged, "outer search did not converge");
    assert!(
        (fit.reml_value - m.reml).abs() < 5e-3,
        "REML {} vs mgcv {}",
        fit.reml_value,
        m.reml
    );
    for (j, (got, want)) in fit.edf_per_term.iter().zip(&m.edf_per_term).enumerate() {
        assert!(
            (got - want).abs() < 0.02,
            "term {j}: edf {got} vs mgcv {want}"
        );
    }
    let eta = fit.predict(x.view()).unwrap();
    let rel = eta
        .iter()
        .zip(&m.fitted)
        .map(|(e, r)| (e.exp() / r - 1.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(rel < 1e-4, "fitted values {rel:.2e} from mgcv's");
}

#[test]
fn nb_lot_like_covariate_lands_on_mgcvs_optimum() {
    let (f, x, y, design) = load(include_str!("fixtures/reml_start_nb_lot_n2000.json"));
    let fit = fit_with_design(negbin_log(1.0), design, x.view(), y.view(), None).unwrap();
    check(&fit, &x, &f.mgcv_output);
}

#[test]
fn nb_house_like_lands_on_mgcvs_optimum() {
    let (f, x, y, design) = load(include_str!("fixtures/reml_start_nb_n1200.json"));
    let fit = fit_with_design(negbin_log(1.0), design, x.view(), y.view(), None).unwrap();
    check(&fit, &x, &f.mgcv_output);
}

#[test]
fn tw_lands_on_mgcvs_optimum_although_it_is_the_worse_one() {
    let (f, x, y, design) = load(include_str!("fixtures/reml_start_tw_n1200.json"));
    let fit = fit_with_design(tweedie_log(1.5, 1.0), design, x.view(), y.view(), None).unwrap();
    check(&fit, &x, &f.mgcv_output);
}
