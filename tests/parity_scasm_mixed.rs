//! Parity with mgcv::scasm for shape-constrained terms next to ordinary
//! ones (cr, re, te with CR margins, numeric by=), Gaussian, on the
//! stage-by-stage fixtures `scasm_internals_{model}_n{n}.json`.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde_json::Value;

use gamrs::design::{MarginKind, TermSpec};
use gamrs::shape::constraints::Shape;
use gamrs::shape::scasm::{fit_fixed_sp, fit_gaussian, ScTermSpec, ScasmDesign};

fn load(name: &str) -> Value {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| v[0].as_f64().unwrap())
}

fn vec_of(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(num).collect(),
        Value::Number(_) => vec![num(v)],
        _ => vec![],
    }
}

fn mat_of(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vec_of).collect();
    let (r, c) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((r, c), rows.into_iter().flatten().collect()).unwrap()
}

fn ints_of(v: &Value) -> Vec<usize> {
    let mut a: Vec<usize> = vec_of(v).into_iter().map(|x| x as usize).collect();
    a.sort();
    a
}

fn max_err(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0, f64::max)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn sc(col: usize) -> ScTermSpec {
    ScTermSpec::Smooth {
        col,
        k: 10,
        shape: Some(Shape::Decreasing),
    }
}

fn cr(col: usize, by: Option<usize>) -> ScTermSpec {
    ScTermSpec::Plain(TermSpec::Cr {
        col,
        k: 8,
        pc: None,
        by,
    })
}

/// Term specs for each fixture model; data columns follow the fixture's
/// `inputs.columns`.
fn specs(model: &str) -> Vec<ScTermSpec> {
    match model {
        "sc_cr" => vec![sc(0), cr(1, None)],
        "sc_cr_re" => vec![
            sc(0),
            cr(1, None),
            ScTermSpec::Plain(TermSpec::Re { col: 2, by: None }),
        ],
        "tesc_cr" => vec![
            ScTermSpec::Tensor {
                cols: [0, 1],
                k: [6, 5],
                shapes: [Some(Shape::Decreasing), Some(Shape::Increasing)],
            },
            cr(2, None),
        ],
        "sc_tecr" => vec![
            sc(0),
            ScTermSpec::Plain(TermSpec::Tensor {
                col_a: 1,
                col_b: 2,
                k_a: 5,
                k_b: 5,
                bs_a: MarginKind::Cr,
                bs_b: MarginKind::Cr,
                by: None,
            }),
        ],
        "sc_crby" => vec![sc(0), cr(1, Some(2))],
        _ => unreachable!(),
    }
}

fn check(model: &str, n: usize) {
    let fx = load(&format!("scasm_internals_{model}_n{n}"));
    let data = mat_of(&fx["inputs"]["x_train"]);
    let y = Array1::from_vec(vec_of(&fx["inputs"]["y_train"]));
    let design = ScasmDesign::build(&data, &specs(model)).unwrap();
    let m = &design.model;
    let setup = &fx["setup"];
    let tag = format!("{model} n={n}");

    // Setup: model matrix, penalties, constraint rows.
    if !setup["X"].is_null() {
        let e = max_err(&m.x, &mat_of(&setup["X"]));
        eprintln!("{tag}: X {e:.2e}");
        assert!(e < 1e-12, "{tag}: X off by {e:.3e}");
    }
    let head = mat_of(&setup["X_head"]);
    let e = max_err(
        &m.x.slice(ndarray::s![0..head.nrows(), ..]).to_owned(),
        &head,
    );
    assert!(e < 1e-12, "{tag}: X head off by {e:.3e}");
    let s_ref: Vec<Array2<f64>> = setup["S"].as_array().unwrap().iter().map(mat_of).collect();
    let mine: Vec<&Array2<f64>> = m.blocks.iter().flat_map(|b| b.penalties.iter()).collect();
    assert_eq!(mine.len(), s_ref.len(), "{tag}: number of penalties");
    let mut e_s = 0.0_f64;
    for (a, b) in mine.iter().zip(&s_ref) {
        let smax = b.iter().map(|v| v.abs()).fold(0.0, f64::max);
        e_s = e_s.max(
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
                / smax,
        );
    }
    let e_c = max_err(&m.constraints.c, &mat_of(&setup["C"]));
    let e_a = max_err(&m.constraints.ain, &mat_of(&setup["Ain"]));
    eprintln!("{tag}: S {e_s:.2e} (rel. to max |S|), C {e_c:.2e}, Ain {e_a:.2e}");
    assert!(
        e_s < 1e-12 && e_c < 1e-12 && e_a < 1e-12,
        "{tag}: setup off"
    );
    let new = mat_of(&fx["inputs"]["x_new"]);
    let e = max_err(
        &design.model_matrix(&new).unwrap(),
        &mat_of(&setup["lpmatrix_new"]),
    );
    eprintln!("{tag}: prediction matrix {e:.2e}");
    assert!(e < 1e-12, "{tag}: prediction matrix off by {e:.3e}");

    // Fixed sp: coefficients and active sets.
    let mut e_fixed = 0.0_f64;
    for rung in fx["fixed_sp"].as_array().unwrap() {
        let sp = Array1::from_vec(vec_of(&rung["sp"]));
        let (beta, active) = fit_fixed_sp(m, &y, &sp).unwrap();
        let b_ref = vec_of(&rung["coefficients"]);
        e_fixed = e_fixed.max(
            beta.iter()
                .zip(&b_ref)
                .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                .fold(0.0, f64::max),
        );
        let mut act: Vec<usize> = active.iter().map(|a| a + 1).collect();
        act.sort();
        assert_eq!(
            act,
            ints_of(&rung["active"]),
            "{tag}: active set at sp {sp}"
        );
    }
    eprintln!("{tag}: fixed-sp coefficients {e_fixed:.2e}");
    assert!(
        e_fixed < 2e-9,
        "{tag}: fixed-sp coefficients off by {e_fixed:.3e}"
    );

    // The EFS loop, iteration by iteration, then the final fit.
    let fit = fit_gaussian(m, &y).unwrap();
    let iters = fx["efs"]["iterations"].as_array().unwrap();
    assert_eq!(
        fit.iterations.len(),
        iters.len(),
        "{tag}: EFS iteration count"
    );
    let mut worst = 0.0_f64;
    let psp0 = vec_of(&iters[0]["pSp"]);
    for (k, (mine, r)) in fit.iterations.iter().zip(iters).enumerate() {
        let mut act: Vec<usize> = mine.active.iter().map(|a| a + 1).collect();
        act.sort();
        assert_eq!(
            act,
            ints_of(&r["active"]),
            "{tag}: active set, iteration {k}"
        );
        for (mv, key) in [
            (&mine.lsp, "lsp"),
            (&mine.tr_s, "trS"),
            (&mine.sedf, "sedf"),
        ] {
            for (a, b) in mv.iter().zip(vec_of(&r[key])) {
                worst = worst.max(rel(*a, b));
            }
        }
        for (j, (a, b)) in mine.psp.iter().zip(vec_of(&r["pSp"])).enumerate() {
            worst = worst.max((a - b).abs() / psp0[j]);
        }
        worst = worst.max(rel(mine.edf_total, r["edf_total"].as_f64().unwrap()));
        worst = worst.max(rel(mine.scale, r["scale"].as_f64().unwrap()));
    }
    let fin = &fx["final"];
    let e_sp = fit
        .sp
        .iter()
        .zip(vec_of(&fin["sp"]))
        .map(|(a, b)| rel(*a, b))
        .fold(0.0, f64::max);
    let e_reml = rel(fit.reml(), fin["reml"].as_f64().unwrap());
    let e_edf = fit
        .edf
        .iter()
        .zip(vec_of(&fin["edf"]))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let e_fit = fit
        .fitted
        .iter()
        .zip(vec_of(&fin["fitted"]))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let e_vp = max_err(&fit.vp, &mat_of(&fin["Vp"]));
    let terms = design.predict_terms(&fit.beta, &new).unwrap();
    let se = design.predict_terms_se(&fit.vp, &new).unwrap();
    let mut e_t = 0.0_f64;
    let mut e_se = 0.0_f64;
    for j in 0..terms.ncols() {
        let tr = vec_of(&fin["term_new"][j]);
        let sr = vec_of(&fin["term_new_se"][j]);
        e_t = e_t.max(
            terms
                .column(j)
                .iter()
                .zip(&tr)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
        );
        e_se = e_se.max(
            se.column(j)
                .iter()
                .zip(&sr)
                .map(|(a, b)| rel(*a, *b))
                .fold(0.0, f64::max),
        );
    }
    eprintln!(
        "{tag}: {} EFS iterations, trace {worst:.2e}; sp {e_sp:.2e}, REML {e_reml:.2e}, edf {e_edf:.2e}, fitted {e_fit:.2e}, Vp {e_vp:.2e}, terms {e_t:.2e}, se {e_se:.2e}",
        iters.len()
    );
    assert!(worst < 5e-8 && e_sp < 5e-8, "{tag}: EFS trace/sp off");
    assert!(
        e_reml < 5e-12 && e_edf < 1e-9 && e_fit < 5e-11 && e_vp < 1e-12,
        "{tag}: final fit off"
    );
    assert!(e_t < 5e-11 && e_se < 5e-10, "{tag}: term predictions off");
}

macro_rules! cases {
    ($($name:ident: $model:expr, $n:expr;)*) => {$(
        #[test]
        fn $name() {
            check($model, $n);
        }
    )*};
}

cases! {
    mixed_sc_cr_n500: "sc_cr", 500;
    mixed_sc_cr_n3000: "sc_cr", 3000;
    mixed_sc_cr_re_n500: "sc_cr_re", 500;
    mixed_sc_cr_re_n3000: "sc_cr_re", 3000;
    mixed_tesc_cr_n500: "tesc_cr", 500;
    mixed_tesc_cr_n3000: "tesc_cr", 3000;
    mixed_sc_tecr_n500: "sc_tecr", 500;
    mixed_sc_tecr_n3000: "sc_tecr", 3000;
    mixed_sc_crby_n500: "sc_crby", 500;
    mixed_sc_crby_n3000: "sc_crby", 3000;
}
