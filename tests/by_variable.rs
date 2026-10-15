//! mgcv's numeric `by=` variable, `s(x, by = w)`: the term contributes
//! `w_i · f(x_i)`.
//!
//! The parity half fits `y ~ s(x, by = w) + s(z)` against mgcv on a 0/1
//! indicator and on a distance-band step (1, 0.6, 0.25, 0)
//! (`scripts/r/gen_by_variable_fixtures.R`). A varying `w` takes the term out
//! of the intercept's span, so mgcv leaves it unconstrained with all `k`
//! coefficients in the raw CR parameterisation, which gamrs shares; that
//! makes the by-term's coefficients directly comparable, not just its fit.
//!
//! The behaviour half pins what parity alone would not: the term is exactly
//! zero where `w = 0`, it scales with `w` on new data (a check that fails if
//! `by` were ignored), a constant `w` falls back to the centred smooth, and
//! the by-term survives persistence.

use std::path::PathBuf;

use ndarray::{s, Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, Predictor, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::{fit_with_design, FittedGam};

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    mgcv_output: MgcvOutput,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    x_new: Vec<Vec<f64>>,
    k: Vec<usize>,
}

#[derive(Deserialize)]
struct MgcvOutput {
    fitted: Vec<f64>,
    coef_by_term: Vec<f64>,
    term_train: Vec<Vec<f64>>,
    term_new: Vec<Vec<f64>>,
    predictions_new: Vec<f64>,
    edf_per_term: Vec<f64>,
    reml: f64,
    scale: f64,
}

fn load(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p).expect("missing fixture");
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn rows(v: &[Vec<f64>]) -> Array2<f64> {
    let ncols = v[0].len();
    Array2::from_shape_vec((v.len(), ncols), v.concat()).unwrap()
}

/// Columns: 0 = x, 1 = z, 2 = w.
fn by_terms(k: &[usize], cr_stable: bool) -> Vec<TermSpec> {
    let (by, plain) = match cr_stable {
        false => (
            TermSpec::Cr {
                col: 0,
                k: k[0],
                pc: None,
                by: Some(2),
            },
            TermSpec::Cr {
                col: 1,
                k: k[1],
                pc: None,
                by: None,
            },
        ),
        true => (
            TermSpec::CrStable {
                col: 0,
                k: k[0],
                pc: None,
                by: Some(2),
            },
            TermSpec::CrStable {
                col: 1,
                k: k[1],
                pc: None,
                by: None,
            },
        ),
    };
    vec![by, plain]
}

fn fit(terms: Vec<TermSpec>, x: &Array2<f64>, y: &Array1<f64>) -> FittedGam {
    fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("fit failed")
}

fn term_contribution(fit: &FittedGam, x_new: &Array2<f64>, j: usize) -> Array1<f64> {
    let Predictor::Additive(additive) = &fit.predictor else {
        panic!("expected an additive predictor");
    };
    let (start, end) = additive.term_col_ranges[j];
    let design = fit.predictor.design(x_new.view()).expect("design");
    design
        .slice(s![.., start..end])
        .dot(&fit.beta.slice(s![start..end]))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

fn assert_parity(name: &str, cr_stable: bool) {
    let fx = load(name);
    let m = &fx.mgcv_output;
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let x_new = rows(&fx.inputs.x_new);
    let f = fit(by_terms(&fx.inputs.k, cr_stable), &x, &y);
    assert!(f.converged);

    let Predictor::Additive(additive) = &f.predictor else {
        panic!("expected an additive predictor");
    };
    let (start, end) = additive.term_col_ranges[0];
    assert_eq!(
        end - start,
        m.coef_by_term.len(),
        "by-term must keep all k columns"
    );

    let fitted = f.predict(x.view()).unwrap();
    let pred_new = f.predict(x_new.view()).unwrap();
    let d_fitted = max_abs_diff(fitted.as_slice().unwrap(), &m.fitted);
    let d_new = max_abs_diff(pred_new.as_slice().unwrap(), &m.predictions_new);
    let d_terms = (0..2)
        .flat_map(|j| {
            [
                max_abs_diff(
                    term_contribution(&f, &x, j).as_slice().unwrap(),
                    &m.term_train[j],
                ),
                max_abs_diff(
                    term_contribution(&f, &x_new, j).as_slice().unwrap(),
                    &m.term_new[j],
                ),
            ]
        })
        .fold(0.0, f64::max);
    let d_edf = max_abs_diff(f.edf_per_term.as_slice().unwrap(), &m.edf_per_term);
    let d_reml = (f.reml_value - m.reml).abs();
    let d_scale = (f.scale - m.scale).abs() / m.scale;
    println!(
        "[{name} stable={cr_stable}] fitted {d_fitted:.2e}  new {d_new:.2e}  terms {d_terms:.2e}  \
         edf {d_edf:.2e}  reml {d_reml:.2e} ({:.6} vs {:.6})  scale rel {d_scale:.2e}",
        f.reml_value, m.reml
    );
    assert!(d_fitted < 1e-5, "fitted values off mgcv by {d_fitted:.3e}");
    assert!(d_new < 1e-5, "new-data predictions off mgcv by {d_new:.3e}");
    assert!(
        d_terms < 1e-5,
        "term contributions off mgcv by {d_terms:.3e}"
    );
    assert!(d_edf < 1e-4, "edf off mgcv by {d_edf:.3e}");
    assert!(d_reml < 1e-6, "REML off mgcv by {d_reml:.3e}");
    assert!(d_scale < 1e-5, "scale off mgcv by {d_scale:.3e}");

    // The unrotated basis is mgcv's own raw CR parameterisation.
    if !cr_stable {
        let beta = f.beta.slice(s![start..end]).to_vec();
        let d_coef = max_abs_diff(&beta, &m.coef_by_term);
        println!("[{name}] by-term coefficients {d_coef:.2e}");
        assert!(
            d_coef < 1e-5,
            "by-term coefficients off mgcv by {d_coef:.3e}"
        );
    }
}

#[test]
fn by_indicator_matches_mgcv() {
    assert_parity("by_indicator_gaussian_n400_k8_cr", false);
    assert_parity("by_indicator_gaussian_n400_k8_cr", true);
}

#[test]
fn by_step_matches_mgcv() {
    assert_parity("by_step_gaussian_n400_k8_cr", false);
    assert_parity("by_step_gaussian_n400_k8_cr", true);
}

#[test]
fn by_term_is_zero_where_w_is_zero_and_scales_with_w() {
    let fx = load("by_step_gaussian_n400_k8_cr");
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let f = fit(by_terms(&fx.inputs.k, false), &x, &y);

    let train_term = term_contribution(&f, &x, 0);
    let zero_rows: Vec<usize> = (0..x.nrows()).filter(|&i| x[[i, 2]] == 0.0).collect();
    assert!(zero_rows.len() > 50);
    assert!(zero_rows.iter().all(|&i| train_term[i] == 0.0));

    // Same (x, z), w doubled then zeroed: the by-term follows w exactly and
    // nothing else moves.
    let x_new = rows(&fx.inputs.x_new);
    let base = term_contribution(&f, &x_new, 0);
    let mut doubled = x_new.clone();
    doubled.column_mut(2).mapv_inplace(|v| 2.0 * v);
    let mut zeroed = x_new.clone();
    zeroed.column_mut(2).fill(0.0);
    let d_doubled = &term_contribution(&f, &doubled, 0) - &(2.0 * &base);
    assert!(d_doubled.iter().all(|d| d.abs() < 1e-12));
    let p_base = f.predict(x_new.view()).unwrap();
    let p_zero = f.predict(zeroed.view()).unwrap();
    let d_zero = &(&p_base - &p_zero) - &base;
    assert!(d_zero.iter().all(|d| d.abs() < 1e-12));
    assert!(base.iter().any(|b| b.abs() > 0.1), "by-term carries signal");
}

/// A constant `w` keeps the centring constraint (mgcv's rule) and spans the
/// plain smooth's model space, so the fit is the plain fit with λ rescaled by
/// `w²`; the tolerance is the outer optimiser's, which lands that λ twice.
#[test]
fn constant_by_falls_back_to_centred_smooth() {
    let fx = load("by_step_gaussian_n400_k8_cr");
    let mut x = rows(&fx.inputs.x_train);
    x.column_mut(2).fill(2.5);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let with_by = fit(by_terms(&fx.inputs.k, false), &x, &y);
    let mut plain_terms = by_terms(&fx.inputs.k, false);
    plain_terms[0] = TermSpec::Cr {
        col: 0,
        k: fx.inputs.k[0],
        pc: None,
        by: None,
    };
    let plain = fit(plain_terms, &x, &y);

    assert_eq!(
        with_by.beta.len(),
        plain.beta.len(),
        "centred: k - 1 columns"
    );
    let d = max_abs_diff(
        with_by.predict(x.view()).unwrap().as_slice().unwrap(),
        plain.predict(x.view()).unwrap().as_slice().unwrap(),
    );
    assert!(
        d < 1e-5,
        "constant-w fit differs from the plain fit by {d:.3e}"
    );
    let d_term = max_abs_diff(
        term_contribution(&with_by, &x, 0).as_slice().unwrap(),
        term_contribution(&plain, &x, 0).as_slice().unwrap(),
    );
    assert!(
        d_term < 1e-5,
        "constant-w term is not the centred smooth: {d_term:.3e}"
    );
}

#[test]
fn by_with_pc_is_refused() {
    let fx = load("by_step_gaussian_n400_k8_cr");
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let mut terms = by_terms(&fx.inputs.k, false);
    terms[0] = TermSpec::Cr {
        col: 0,
        k: 8,
        pc: Some(0.5),
        by: Some(2),
    };
    let err = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .err()
    .expect("pc with by must be refused");
    assert!(err.to_string().contains("pc or by"), "{err}");
}

/// `∂(w·f(x))/∂x = w·f'(x)` and `∂/∂w = f(x)`, against central differences.
#[test]
fn by_term_derivatives_match_finite_differences() {
    let fx = load("by_step_gaussian_n400_k8_cr");
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let f = fit(by_terms(&fx.inputs.k, false), &x, &y);
    let x_new = rows(&fx.inputs.x_new);
    let h = 1e-6;
    for axis in [0, 2] {
        let analytic = f
            .predictor
            .design_deriv(x_new.view(), axis)
            .unwrap()
            .dot(&f.beta);
        let (mut up, mut down) = (x_new.clone(), x_new.clone());
        up.column_mut(axis).mapv_inplace(|v| v + h);
        down.column_mut(axis).mapv_inplace(|v| v - h);
        let fd = (&f.predict(up.view()).unwrap() - &f.predict(down.view()).unwrap()) / (2.0 * h);
        let d = max_abs_diff(analytic.as_slice().unwrap(), fd.as_slice().unwrap());
        assert!(
            d < 1e-5,
            "axis {axis}: derivative off finite differences by {d:.3e}"
        );
    }
}

#[test]
fn by_term_survives_persistence() {
    let fx = load("by_indicator_gaussian_n400_k8_cr");
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let x_new = rows(&fx.inputs.x_new);
    let f = fit(by_terms(&fx.inputs.k, false), &x, &y);
    let expected = f.predict(x_new.view()).unwrap();
    let from_bincode = FittedGam::deserialize(&f.serialize().unwrap()).unwrap();
    let from_json = FittedGam::deserialize_json(&f.serialize_json().unwrap()).unwrap();
    assert_eq!(from_bincode.predict(x_new.view()).unwrap(), expected);
    let d = &from_json.predict(x_new.view()).unwrap() - &expected;
    assert!(d.iter().all(|d| d.abs() < 1e-15));
}
