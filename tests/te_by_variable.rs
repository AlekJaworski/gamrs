//! mgcv's numeric `by=` on a tensor product, `te(x, z, by = w)`: the term
//! contributes `w_i · f(x_i, z_i)`.
//!
//! Parity against mgcv on two fits (`scripts/r/gen_te_by_fixtures.R`):
//! `te(x, z, by = x)` with x >= 0, which pins the surface to 0 along x = 0,
//! and `te(x, z, by = w) + s(v)` with w a covariate of its own. A varying `w`
//! leaves the tensor unconstrained, so it keeps all `k_a · k_b` coefficients.

use std::path::PathBuf;

use ndarray::{s, Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, MarginKind, Predictor, TermSpec};
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
    edf_per_term: Vec<f64>,
    n_coef_per_term: Vec<usize>,
    reml: f64,
    scale: f64,
}

const SELF: &str = "te_by_self_gaussian_n400_k4x3_cr";
const COVARIATE: &str = "te_by_covariate_gaussian_n400_k4x3_cr";

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

fn te(by: Option<usize>) -> TermSpec {
    TermSpec::Tensor {
        col_a: 0,
        col_b: 1,
        k_a: 4,
        k_b: 3,
        bs_a: MarginKind::Cr,
        bs_b: MarginKind::Cr,
        by,
    }
}

/// The fixture's model. SELF columns: 0 = x, 1 = z. COVARIATE columns:
/// 0 = x, 1 = z, 2 = w, 3 = v.
fn terms(name: &str) -> (Vec<TermSpec>, usize) {
    match name {
        SELF => (vec![te(Some(0))], 0),
        COVARIATE => (
            vec![
                te(Some(2)),
                TermSpec::Cr {
                    col: 3,
                    k: 8,
                    pc: None,
                    by: None,
                },
            ],
            2,
        ),
        _ => unreachable!(),
    }
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

fn assert_parity(name: &str) {
    let fx = load(name);
    let m = &fx.mgcv_output;
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let x_new = rows(&fx.inputs.x_new);
    let (terms, by_col) = terms(name);
    let f = fit(terms, &x, &y);
    assert!(f.converged);

    let Predictor::Additive(additive) = &f.predictor else {
        panic!("expected an additive predictor");
    };
    let widths: Vec<usize> = additive
        .term_col_ranges
        .iter()
        .map(|(a, b)| b - a)
        .collect();
    assert_eq!(
        widths, m.n_coef_per_term,
        "by-tensor keeps all k_a * k_b columns"
    );

    let fitted = f.predict(x.view()).unwrap();
    let pred_new = f.predict(x_new.view()).unwrap();
    let d_fitted = max_abs_diff(fitted.as_slice().unwrap(), &m.fitted);
    let d_new = max_abs_diff(pred_new.as_slice().unwrap(), &m.predictions_new);
    let d_terms = (0..widths.len())
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
    // `edf_per_term` has one entry per penalty; the by-tensor holds the first
    // two and each other term one, so the tensor's edf is what is left.
    let other_edf = f.edf_per_term.slice(s![2..]).to_vec();
    let edf: Vec<f64> = std::iter::once(f.edf_total - 1.0 - other_edf.iter().sum::<f64>())
        .chain(other_edf)
        .collect();
    let d_edf = max_abs_diff(&edf, &m.edf_per_term);
    let d_reml = (f.reml_value - m.reml).abs();
    let d_scale = (f.scale - m.scale).abs() / m.scale;
    println!(
        "[{name}] fitted {d_fitted:.2e}  new {d_new:.2e}  terms {d_terms:.2e}  \
         edf {d_edf:.2e} ({:?} vs {:?})  reml {d_reml:.2e} ({:.6} vs {:.6})  scale rel {d_scale:.2e}",
        edf,
        m.edf_per_term,
        f.reml_value,
        m.reml
    );
    // Both fits send one margin's λ along a ridge towards infinity; gamrs
    // stops a few 1e-6 of REML short of mgcv's tightly converged end, which
    // is what these bounds absorb.
    assert!(d_fitted < 5e-5, "fitted values off mgcv by {d_fitted:.3e}");
    assert!(d_new < 5e-5, "new-data predictions off mgcv by {d_new:.3e}");
    assert!(
        d_terms < 5e-5,
        "term contributions off mgcv by {d_terms:.3e}"
    );
    assert!(d_edf < 1e-3, "edf off mgcv by {d_edf:.3e}");
    assert!(d_reml < 1e-5, "REML off mgcv by {d_reml:.3e}");
    assert!(d_scale < 1e-5, "scale off mgcv by {d_scale:.3e}");

    for (xs, label) in [(&x, "train"), (&x_new, "new")] {
        let term = term_contribution(&f, xs, 0);
        let zero_rows: Vec<usize> = (0..xs.nrows())
            .filter(|&i| xs[[i, by_col]] == 0.0)
            .collect();
        assert!(!zero_rows.is_empty());
        assert!(
            zero_rows.iter().all(|&i| term[i] == 0.0),
            "{label}: the by-tensor must be exactly 0 where w = 0"
        );
    }
}

#[test]
fn te_by_self_matches_mgcv() {
    assert_parity(SELF);
}

#[test]
fn te_by_covariate_matches_mgcv() {
    assert_parity(COVARIATE);
}

/// A constant `w` keeps the sum-to-zero constraint and fits the plain
/// tensor, with λ rescaled by `w²`. On the plain `te` parity data, where no
/// λ runs off along a ridge, so both fits land on the same optimum.
#[test]
fn constant_by_falls_back_to_centred_tensor() {
    #[derive(Deserialize)]
    struct Plain {
        inputs: PlainInputs,
    }
    #[derive(Deserialize)]
    struct PlainInputs {
        x_train: Vec<Vec<f64>>,
        y_train: Vec<f64>,
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/2d_gaussian_te_n300_k5x5.json");
    let fx: Plain = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let xz = rows(&fx.inputs.x_train);
    let mut x = Array2::from_elem((xz.nrows(), 3), 2.5);
    x.slice_mut(s![.., ..2]).assign(&xz);
    let y = Array1::from_vec(fx.inputs.y_train);
    let tensor = |by| TermSpec::Tensor {
        col_a: 0,
        col_b: 1,
        k_a: 5,
        k_b: 5,
        bs_a: MarginKind::Cr,
        bs_b: MarginKind::Cr,
        by,
    };
    let with_by = fit(vec![tensor(Some(2))], &x, &y);
    let plain = fit(vec![tensor(None)], &x, &y);
    assert_eq!(
        with_by.beta.len(),
        plain.beta.len(),
        "centred: k_a * k_b - 1 columns"
    );
    let d = max_abs_diff(
        with_by.predict(x.view()).unwrap().as_slice().unwrap(),
        plain.predict(x.view()).unwrap().as_slice().unwrap(),
    );
    assert!(
        d < 1e-6,
        "constant-w fit differs from the plain fit by {d:.3e}"
    );
}

/// `∂(w·f)/∂x = w·∂f/∂x (+ f when w is x)`, against central differences, on
/// every column: the margins, the by column and one the tensor ignores.
#[test]
fn by_tensor_derivatives_match_finite_differences() {
    for name in [SELF, COVARIATE] {
        let fx = load(name);
        let x = rows(&fx.inputs.x_train);
        let y = Array1::from_vec(fx.inputs.y_train.clone());
        let (terms, _) = terms(name);
        let f = fit(terms, &x, &y);
        // Off the knots and the x = 0 boundary, so central differences are smooth.
        let x_new = rows(&fx.inputs.x_new).mapv(|v| v + 0.013);
        for axis in 0..x_new.ncols() {
            let h = 1e-6
                * x_new
                    .column(axis)
                    .iter()
                    .fold(1.0_f64, |a, v| a.max(v.abs()));
            let analytic = f
                .predictor
                .design_deriv(x_new.view(), axis)
                .unwrap()
                .dot(&f.beta);
            let (mut up, mut down) = (x_new.clone(), x_new.clone());
            up.column_mut(axis).mapv_inplace(|v| v + h);
            down.column_mut(axis).mapv_inplace(|v| v - h);
            let fd =
                (&f.predict(up.view()).unwrap() - &f.predict(down.view()).unwrap()) / (2.0 * h);
            let scale = fd.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
            let d = max_abs_diff(analytic.as_slice().unwrap(), fd.as_slice().unwrap()) / scale;
            assert!(
                d < 1e-5,
                "[{name}] axis {axis}: derivative off finite differences by {d:.3e} (relative)"
            );
        }
    }
}

#[test]
fn by_tensor_survives_persistence() {
    let fx = load(COVARIATE);
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let x_new = rows(&fx.inputs.x_new);
    let (terms, _) = terms(COVARIATE);
    let f = fit(terms, &x, &y);
    let expected = f.predict(x_new.view()).unwrap();
    let from_bincode = FittedGam::deserialize(&f.serialize().unwrap()).unwrap();
    let from_json = FittedGam::deserialize_json(&f.serialize_json().unwrap()).unwrap();
    assert_eq!(from_bincode.predict(x_new.view()).unwrap(), expected);
    let d = &from_json.predict(x_new.view()).unwrap() - &expected;
    assert!(d.iter().all(|d| d.abs() < 1e-15));
}
