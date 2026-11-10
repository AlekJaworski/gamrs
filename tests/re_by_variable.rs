//! mgcv's numeric `by=` on a random effect, `s(g, by = w, bs = "re")`: the
//! term contributes `w_i · b_{g_i}`.
//!
//! The parity half fits `y ~ golf + s(course, by = golf, bs = "re") + s(x)`
//! with every non-golf row on a placeholder course, and a random slope
//! `y ~ w + s(g, by = w, bs = "re") + s(x)` with one group seen only where
//! `w = 0` (`scripts/r/gen_re_by_fixtures.R`). A random effect carries no
//! constraint, so mgcv keeps one column per level, including the levels
//! seen only where `w = 0`: those columns are all zero, and the identity
//! penalty holds their coefficients at exactly 0. gamrs does the same, so
//! the random-effect coefficients are directly comparable.
//!
//! The behaviour half pins what parity alone would not: the term is exactly
//! zero where `w = 0`, it scales with `w` on new data, a constant `w` is the
//! plain random effect with λ rescaled, and the by-term survives persistence.

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
}

#[derive(Deserialize)]
struct MgcvOutput {
    fitted: Vec<f64>,
    term_train: Vec<Vec<f64>>,
    term_new: Vec<Vec<f64>>,
    predictions_new: Vec<f64>,
    coef_parametric: Vec<f64>,
    coef_re: Vec<f64>,
    edf_per_term: Vec<f64>,
    reml: f64,
    scale: f64,
    sp: Vec<f64>,
}

const GOLF: &str = "re_by_golf_gaussian_n400_cr8";
const SLOPE: &str = "re_by_slope_gaussian_n400_cr8";

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

/// Columns: 0 = w (golf), 1 = group (course), 2 = x. Term order is mgcv's
/// coefficient order: w, then the random effect, then s(x).
fn re_by_terms(by: Option<usize>) -> Vec<TermSpec> {
    vec![
        TermSpec::Parametric { col: 0 },
        TermSpec::Re { col: 1, by },
        TermSpec::Cr {
            col: 2,
            k: 8,
            pc: None,
            by: None,
        },
    ]
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

fn term_range(fit: &FittedGam, j: usize) -> (usize, usize) {
    let Predictor::Additive(additive) = &fit.predictor else {
        panic!("expected an additive predictor");
    };
    additive.term_col_ranges[j]
}

fn term_contribution(fit: &FittedGam, x_new: &Array2<f64>, j: usize) -> Array1<f64> {
    let (start, end) = term_range(fit, j);
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

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs() / q.abs())
        .fold(0.0, f64::max)
}

fn assert_parity(name: &str) {
    let fx = load(name);
    let m = &fx.mgcv_output;
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let x_new = rows(&fx.inputs.x_new);
    let f = fit(re_by_terms(Some(0)), &x, &y);
    assert!(f.converged);

    let (start, end) = term_range(&f, 1);
    assert_eq!(end - start, m.coef_re.len(), "every level keeps its column");

    let fitted = f.predict(x.view()).unwrap();
    let pred_new = f.predict(x_new.view()).unwrap();
    let d_fitted = max_abs_diff(fitted.as_slice().unwrap(), &m.fitted);
    let d_new = max_abs_diff(pred_new.as_slice().unwrap(), &m.predictions_new);
    let d_terms = (0..2)
        .flat_map(|j| {
            [
                max_abs_diff(
                    term_contribution(&f, &x, j + 1).as_slice().unwrap(),
                    &m.term_train[j],
                ),
                max_abs_diff(
                    term_contribution(&f, &x_new, j + 1).as_slice().unwrap(),
                    &m.term_new[j],
                ),
            ]
        })
        .fold(0.0, f64::max);
    let d_coef = max_abs_diff(f.beta.slice(s![start..end]).as_slice().unwrap(), &m.coef_re);
    let d_param = max_abs_diff(
        f.beta.slice(s![0..2]).as_slice().unwrap(),
        &m.coef_parametric,
    );
    let d_edf = max_abs_diff(f.edf_per_term.as_slice().unwrap(), &m.edf_per_term);
    let d_reml = (f.reml_value - m.reml).abs();
    let d_scale = (f.scale - m.scale).abs() / m.scale;
    let d_sp = max_rel_diff(f.lambda.as_slice().unwrap(), &m.sp);
    println!(
        "[{name}] fitted {d_fitted:.2e}  new {d_new:.2e}  terms {d_terms:.2e}  re coef \
         {d_coef:.2e}  parametric {d_param:.2e}  edf {d_edf:.2e}  reml {d_reml:.2e} \
         ({:.6} vs {:.6})  scale rel {d_scale:.2e}  sp rel {d_sp:.2e}",
        f.reml_value, m.reml
    );
    assert!(d_fitted < 1e-5, "fitted values off mgcv by {d_fitted:.3e}");
    assert!(d_new < 1e-5, "new-data predictions off mgcv by {d_new:.3e}");
    assert!(
        d_terms < 1e-5,
        "term contributions off mgcv by {d_terms:.3e}"
    );
    assert!(
        d_coef < 1e-5,
        "random-effect coefficients off mgcv by {d_coef:.3e}"
    );
    assert!(
        d_param < 1e-5,
        "parametric coefficients off mgcv by {d_param:.3e}"
    );
    assert!(d_edf < 1e-4, "edf off mgcv by {d_edf:.3e}");
    assert!(d_reml < 1e-6, "REML off mgcv by {d_reml:.3e}");
    assert!(d_scale < 1e-5, "scale off mgcv by {d_scale:.3e}");
    assert!(d_sp < 1e-3, "sp off mgcv by {d_sp:.3e} (relative)");
}

#[test]
fn re_by_golf_indicator_matches_mgcv() {
    assert_parity(GOLF);
}

#[test]
fn re_by_slope_matches_mgcv() {
    assert_parity(SLOPE);
}

/// A level seen only where `w = 0` (the golf placeholder course, the slope
/// fixture's group 6) has an all-zero column; the identity penalty pins its
/// coefficient to 0 rather than leaving it unidentified.
#[test]
fn a_level_seen_only_at_w_zero_is_fitted_as_zero() {
    for (name, level) in [(GOLF, 0usize), (SLOPE, 5)] {
        let fx = load(name);
        let x = rows(&fx.inputs.x_train);
        let y = Array1::from_vec(fx.inputs.y_train.clone());
        let f = fit(re_by_terms(Some(0)), &x, &y);
        let (start, _) = term_range(&f, 1);
        let b = f.beta[start + level];
        assert!(b.abs() < 1e-12, "[{name}] coefficient {b:.3e}, expected 0");
        assert!(f.beta.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn re_by_term_is_zero_where_w_is_zero_and_scales_with_w() {
    let fx = load(SLOPE);
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let f = fit(re_by_terms(Some(0)), &x, &y);

    let train_term = term_contribution(&f, &x, 1);
    let zero_rows: Vec<usize> = (0..x.nrows()).filter(|&i| x[[i, 0]] == 0.0).collect();
    assert!(zero_rows.len() > 50);
    assert!(zero_rows.iter().all(|&i| train_term[i] == 0.0));

    let x_new = rows(&fx.inputs.x_new);
    let base = term_contribution(&f, &x_new, 1);
    let mut doubled = x_new.clone();
    doubled.column_mut(0).mapv_inplace(|v| 2.0 * v);
    let d_doubled = &term_contribution(&f, &doubled, 1) - &(2.0 * &base);
    assert!(d_doubled.iter().all(|d| d.abs() < 1e-12));
    assert!(base.iter().any(|b| b.abs() > 0.1), "by-term carries signal");
}

/// A constant `w = c` spans the plain random effect's model space, so the fit
/// is the plain fit with λ rescaled by `c²`; the tolerance is the outer
/// optimiser's, which lands that λ twice.
#[test]
fn constant_by_is_the_plain_random_effect() {
    let fx = load(GOLF);
    let mut x = rows(&fx.inputs.x_train);
    x.column_mut(0).fill(2.5);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let mut with_by_terms = re_by_terms(Some(0));
    let mut plain_terms = re_by_terms(None);
    // A constant w would alias the parametric term with the intercept.
    with_by_terms.remove(0);
    plain_terms.remove(0);
    let with_by = fit(with_by_terms, &x, &y);
    let plain = fit(plain_terms, &x, &y);

    assert_eq!(with_by.beta.len(), plain.beta.len());
    let d = max_abs_diff(
        with_by.predict(x.view()).unwrap().as_slice().unwrap(),
        plain.predict(x.view()).unwrap().as_slice().unwrap(),
    );
    assert!(
        d < 1e-5,
        "constant-w fit differs from the plain fit by {d:.3e}"
    );
    let ratio = with_by.lambda[0] / plain.lambda[0];
    assert!(
        (ratio / 2.5f64.powi(2) - 1.0).abs() < 1e-3,
        "λ ratio {ratio:.6}, expected c² = 6.25"
    );
}

/// `∂(w·b_g)/∂g = 0` and `∂/∂w = b_g`, against central differences.
#[test]
fn re_by_term_derivatives() {
    let fx = load(SLOPE);
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let f = fit(re_by_terms(Some(0)), &x, &y);
    let x_new = rows(&fx.inputs.x_new);
    let (start, end) = term_range(&f, 1);

    let d_group = f.predictor.design_deriv(x_new.view(), 1).unwrap();
    assert!(d_group.slice(s![.., start..end]).iter().all(|v| *v == 0.0));

    let h = 1e-6;
    let analytic = f
        .predictor
        .design_deriv(x_new.view(), 0)
        .unwrap()
        .dot(&f.beta);
    let (mut up, mut down) = (x_new.clone(), x_new.clone());
    up.column_mut(0).mapv_inplace(|v| v + h);
    down.column_mut(0).mapv_inplace(|v| v - h);
    let fd = (&f.predict(up.view()).unwrap() - &f.predict(down.view()).unwrap()) / (2.0 * h);
    let d = max_abs_diff(analytic.as_slice().unwrap(), fd.as_slice().unwrap());
    assert!(d < 1e-6, "∂/∂w off finite differences by {d:.3e}");
}

#[test]
fn re_by_term_survives_persistence() {
    let fx = load(GOLF);
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let x_new = rows(&fx.inputs.x_new);
    let f = fit(re_by_terms(Some(0)), &x, &y);
    let expected = f.predict(x_new.view()).unwrap();
    let from_bincode = FittedGam::deserialize(&f.serialize().unwrap()).unwrap();
    let from_json = FittedGam::deserialize_json(&f.serialize_json().unwrap()).unwrap();
    assert_eq!(from_bincode.predict(x_new.view()).unwrap(), expected);
    let d = &from_json.predict(x_new.view()).unwrap() - &expected;
    assert!(d.iter().all(|d| d.abs() < 1e-15));
}
