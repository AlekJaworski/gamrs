//! Parity with mgcv::scasm (Gaussian, shape-constrained `sc` smooths), stage
//! by stage, on the fixtures from `scripts/r/gen_scasm_internals_fixtures.R`.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde_json::Value;

use gamrs::shape::bspline::{column_means, scale_penalty, BsBasis};

fn load(name: &str) -> Value {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

/// A row that R's JSON writer may have unboxed to a scalar.
fn first(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| v[0].as_f64().unwrap())
}

fn vec_of(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn mat_of(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vec_of).collect();
    let (r, c) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((r, c), rows.into_iter().flatten().collect()).unwrap()
}

/// Largest |a - b| / (1 + |b|).
fn max_err(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0, f64::max)
}

struct Case {
    internals: Value,
    x: Array1<f64>,
}

fn case_1d(n: usize) -> Case {
    let internals = load(&format!("scasm_internals_mono_n{n}_k10"));
    let src = load(&format!("scasm_mono_gaussian_n{n}_k10"));
    let x: Array1<f64> = src["inputs"]["x_train"]
        .as_array()
        .unwrap()
        .iter()
        .map(first)
        .collect();
    Case { internals, x }
}

fn with_intercept(b: &Array2<f64>) -> Array2<f64> {
    let mut x = Array2::<f64>::ones((b.nrows(), b.ncols() + 1));
    x.slice_mut(ndarray::s![.., 1..]).assign(b);
    x
}

fn check_setup_1d(n: usize) {
    let c = case_1d(n);
    let setup = &c.internals["setup"];
    let sm = &setup["smooths"][0];
    let basis = BsBasis::from_data(c.x.view(), 10, (3, 2)).unwrap();

    let knots = vec_of(&sm["knots"]);
    let knot_err = basis
        .knots
        .iter()
        .zip(&knots)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(knot_err < 1e-13, "knots off by {knot_err:.3e}");

    let xb = basis.design(c.x.view());
    let x = with_intercept(&xb);
    if !setup["X"].is_null() {
        let e = max_err(&x, &mat_of(&setup["X"]));
        assert!(e < 1e-12, "X off by {e:.3e}");
        eprintln!("n={n}: X max error {e:.2e}");
    }
    let head = mat_of(&setup["X_head"]);
    let e = max_err(&x.slice(ndarray::s![0..head.nrows(), ..]).to_owned(), &head);
    assert!(e < 1e-12, "X head off by {e:.3e}");
    let cm: Array2<f64> = column_means(&x).insert_axis(ndarray::Axis(0));
    let e = max_err(
        &cm,
        &Array2::from_shape_vec((1, x.ncols()), vec_of(&setup["X_colmeans"])).unwrap(),
    );
    assert!(e < 1e-12, "X column means off by {e:.3e}");

    // C is the smooth block's column means, 0 on the intercept.
    let mut c_row = column_means(&x);
    c_row[0] = 0.0;
    let e = max_err(&c_row.insert_axis(ndarray::Axis(0)), &mat_of(&setup["C"]));
    assert!(e < 1e-12, "C off by {e:.3e}");

    let (s, scale) = scale_penalty(&basis.penalty(), &xb);
    let s_ref = mat_of(&setup["S"][0]);
    let s_max = s_ref.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let e = s
        .iter()
        .zip(s_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        / s_max;
    assert!(e < 1e-12, "S off by {e:.3e} (relative to max |S|)");
    eprintln!("n={n}: S max error {e:.2e} (relative to max |S|)");
    let scale_ref = sm["S_scale"][0].as_f64().unwrap();
    assert!(
        (scale - scale_ref).abs() / scale_ref < 1e-12,
        "penalty scale {scale} vs {scale_ref}"
    );

    // Prediction matrix on new points, two of them outside the data range.
    let xn: Array1<f64> = c.internals["inputs"]["x_new"]
        .as_array()
        .unwrap()
        .iter()
        .map(first)
        .collect();
    let e = max_err(
        &with_intercept(&basis.design(xn.view())),
        &mat_of(&setup["lpmatrix_new"]),
    );
    assert!(e < 1e-12, "prediction matrix off by {e:.3e}");
    eprintln!("n={n}: prediction matrix max error {e:.2e}");
}

#[test]
fn scasm_setup_1d_n500() {
    check_setup_1d(500);
}

#[test]
fn scasm_setup_1d_n3000() {
    check_setup_1d(3000);
}

fn check_constraints_1d(n: usize) {
    use gamrs::shape::constraints::{monotone_start, shape_rows, Constraints, Shape};
    let c = case_1d(n);
    let setup = &c.internals["setup"];
    let basis = BsBasis::from_data(c.x.view(), 10, (3, 2)).unwrap();
    let xb = basis.design(c.x.view());
    let cm = column_means(&xb);
    let mut cons = Constraints::empty(11);
    cons.add_smooth(1, &shape_rows(Shape::Decreasing, 10), Some(&cm));
    assert_eq!(cons.ain, mat_of(&setup["Ain"]), "Ain differs");
    assert_eq!(cons.bin.to_vec(), vec_of(&setup["bin"]), "bin differs");
    let e = max_err(&cons.c, &mat_of(&setup["C"]));
    assert!(e < 1e-12, "C off by {e:.3e}");

    let mut beta0 = Array1::<f64>::zeros(11);
    beta0
        .slice_mut(ndarray::s![1..])
        .assign(&monotone_start(Shape::Decreasing, &cm));
    let b_ref = vec_of(&setup["beta0"]);
    let e = beta0
        .iter()
        .zip(&b_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("n={n}: feasible start max error {e:.2e}");
    assert!(e < 1e-15, "beta0 off by {e:.3e}");
}

#[test]
fn scasm_constraints_1d_n500() {
    check_constraints_1d(500);
}

#[test]
fn scasm_constraints_1d_n3000() {
    check_constraints_1d(3000);
}

fn check_fixed_sp_1d(n: usize) {
    use gamrs::shape::constraints::{monotone_start, shape_rows, Constraints, Shape};
    use gamrs::shape::pcls::{normal_equations, pcls};
    let c = case_1d(n);
    let src = load(&format!("scasm_mono_gaussian_n{n}_k10"));
    let y = Array1::from_vec(vec_of(&src["inputs"]["y_train"]));
    let basis = BsBasis::from_data(c.x.view(), 10, (3, 2)).unwrap();
    let xb = basis.design(c.x.view());
    let x = with_intercept(&xb);
    let cm = column_means(&xb);
    let (s, _) = scale_penalty(&basis.penalty(), &xb);
    let mut cons = Constraints::empty(11);
    cons.add_smooth(1, &shape_rows(Shape::Decreasing, 10), Some(&cm));
    let mut beta0 = Array1::<f64>::zeros(11);
    beta0
        .slice_mut(ndarray::s![1..])
        .assign(&monotone_start(Shape::Decreasing, &cm));
    let w = Array1::<f64>::ones(n);
    for rung in c.internals["fixed_sp"].as_array().unwrap() {
        let sp = first(&rung["sp"]);
        let (h, xty) = normal_equations(&x, &w, &y, &[(1, &s, sp)]);
        let fit = pcls(&h, &xty, &cons.ain, &cons.bin, &cons.c, &beta0).unwrap();
        let b_ref = vec_of(&rung["coefficients"]);
        let e = fit
            .beta
            .iter()
            .zip(&b_ref)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0, f64::max);
        let act: Vec<i64> = fit.active.iter().map(|a| *a as i64 + 1).collect();
        let act_ref: Vec<i64> = match &rung["active"] {
            Value::Array(v) => v.iter().map(|a| a.as_i64().unwrap()).collect(),
            Value::Number(a) => vec![a.as_i64().unwrap()],
            _ => vec![],
        };
        eprintln!("n={n} sp={sp:.4e}: coef max error {e:.2e}, active {act:?} (scasm {act_ref:?})");
        assert!(e < 2e-12, "coefficients off by {e:.3e} at sp {sp}");
        assert_eq!(act, act_ref, "active set differs at sp {sp}");
    }
}

#[test]
fn scasm_fixed_sp_1d_n500() {
    check_fixed_sp_1d(500);
}

#[test]
fn scasm_fixed_sp_1d_n3000() {
    check_fixed_sp_1d(3000);
}

fn ints_of(v: &Value) -> Vec<usize> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_u64().unwrap() as usize).collect(),
        Value::Number(x) => vec![x.as_u64().unwrap() as usize],
        _ => vec![],
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn check_efs_1d(n: usize) {
    use gamrs::shape::constraints::Shape;
    use gamrs::shape::scasm::{fit_gaussian, ScSpec, ScasmDesign};
    let c = case_1d(n);
    let src = load(&format!("scasm_mono_gaussian_n{n}_k10"));
    let y = Array1::from_vec(vec_of(&src["inputs"]["y_train"]));
    let data = c.x.clone().insert_axis(ndarray::Axis(1));
    let design = ScasmDesign::build(
        &data,
        &[ScSpec {
            col: 0,
            k: 10,
            shape: Some(Shape::Decreasing),
        }],
    )
    .unwrap();
    let fit = fit_gaussian(&design.model, &y).unwrap();

    // Every EFS iteration against the traced R loop.
    let efs = &c.internals["efs"];
    let lsp0 = first(&efs["lsp_initial"]);
    let iters = efs["iterations"].as_array().unwrap();
    assert_eq!(fit.iterations.len(), iters.len(), "EFS iteration count");
    let mut worst = 0.0_f64;
    for (k, (mine, r)) in fit.iterations.iter().zip(iters).enumerate() {
        if k == 0 {
            worst = worst.max(rel(mine.lsp[0], lsp0));
        }
        assert_eq!(
            mine.active.iter().map(|a| a + 1).collect::<Vec<_>>(),
            ints_of(&r["active"]),
            "active set, iteration {k}"
        );
        let rank = (mine.tr_s[0] * mine.lsp[0].exp()).round();
        let rank_ref = (first(&r["trS"]) * first(&r["lsp"]).exp()).round();
        assert_eq!(rank, rank_ref, "projected penalty rank, iteration {k}");
        for (a, b) in [
            (mine.lsp[0], first(&r["lsp"])),
            (mine.edf_total, r["edf_total"].as_f64().unwrap()),
            (mine.scale, r["scale"].as_f64().unwrap()),
            (mine.sedf[0], first(&r["sedf"])),
            (mine.tr_s[0], first(&r["trS"])),
            (mine.psp[0], first(&r["pSp"])),
        ] {
            worst = worst.max(rel(a, b));
        }
    }
    eprintln!(
        "n={n}: {} EFS iterations, worst relative error over the trace {worst:.2e}",
        iters.len()
    );
    assert!(worst < 1e-11, "EFS trace off by {worst:.3e}");

    let fin = &c.internals["final"];
    let e_sp = rel(fit.sp[0], first(&fin["sp"]));
    let e_reml = rel(fit.reml(), fin["reml"].as_f64().unwrap());
    let e_scale = rel(fit.scale, fin["scale"].as_f64().unwrap());
    let edf_ref = vec_of(&fin["edf"]);
    let e_edf = fit
        .edf
        .iter()
        .zip(&edf_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let fitted_ref = vec_of(&fin["fitted"]);
    let e_fit = fit
        .fitted
        .iter()
        .zip(&fitted_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let coef_ref = vec_of(&fin["coefficients"]);
    let e_coef = fit
        .beta
        .iter()
        .zip(&coef_ref)
        .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
        .fold(0.0, f64::max);
    let e_vp = max_err(&fit.vp, &mat_of(&fin["Vp"]));
    eprintln!(
        "n={n}: sp {e_sp:.2e} rel, REML {e_reml:.2e} rel, scale {e_scale:.2e} rel, edf {e_edf:.2e} abs, fitted {e_fit:.2e} abs, coef {e_coef:.2e}, Vp {e_vp:.2e}"
    );
    assert!(
        e_sp < 1e-11 && e_reml < 1e-13 && e_scale < 1e-13,
        "sp/REML/scale off"
    );
    assert!(
        e_edf < 1e-12 && e_fit < 1e-12 && e_coef < 1e-12 && e_vp < 1e-12,
        "edf/fitted/coef/Vp off"
    );

    // predict(type = "terms") and its standard errors on new points.
    let xn: Array1<f64> = c.internals["inputs"]["x_new"]
        .as_array()
        .unwrap()
        .iter()
        .map(first)
        .collect();
    let new = xn.insert_axis(ndarray::Axis(1));
    let terms = design.predict_terms(&fit.beta, &new);
    let se = design.predict_terms_se(&fit.vp, &new);
    let t_ref = vec_of(&fin["term_new"][0]);
    let se_ref = vec_of(&fin["term_new_se"][0]);
    let e_t = terms
        .column(0)
        .iter()
        .zip(&t_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let e_se = se
        .column(0)
        .iter()
        .zip(&se_ref)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    eprintln!("n={n}: terms on new data {e_t:.2e} abs, their se {e_se:.2e} rel");
    assert!(e_t < 1e-12 && e_se < 1e-11, "predicted terms off");

    // Monotone by construction: dense grid across and beyond the data.
    let (lo, hi) = (
        c.x.fold(f64::INFINITY, |a, &b| a.min(b)),
        c.x.fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
    );
    let grid: Array1<f64> = Array1::linspace(lo - 1.0, hi + 1.0, 5001);
    let f = design.predict_terms(&fit.beta, &grid.insert_axis(ndarray::Axis(1)));
    let rise = f
        .column(0)
        .windows(2)
        .into_iter()
        .map(|w| w[1] - w[0])
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!("n={n}: largest increase between grid neighbours {rise:.2e}");
    assert!(
        rise <= 1e-12,
        "decreasing smooth increases by {rise:.3e} on the grid"
    );
}

#[test]
fn scasm_efs_1d_n500() {
    check_efs_1d(500);
}

#[test]
fn scasm_efs_1d_n3000() {
    check_efs_1d(3000);
}
