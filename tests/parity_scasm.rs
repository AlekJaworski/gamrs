//! Parity with mgcv::scasm for shape-constrained (`sc`) terms, alone and
//! next to ordinary cr, re, te(cr) and by= terms, Gaussian, on the
//! fixtures `scasm_internals_*.json` (built by
//! `scripts/r/gen_scasm_internals_fixtures.R` on synthetic data).
//!
//! gamrs absorbs each term's centring into its basis; scasm keeps it as an
//! equality row. The two coefficient vectors are related by the
//! block-diagonal centring map `Z` (`β_scasm = Z β_gamrs`), which the
//! checks read off the fitted predictors. Tensor terms use gamrs's te
//! penalty convention (no te-level `scale.penalty`), so their λ differ from
//! mgcv's by a constant per penalty; the checks measure that constant from
//! the penalties and assert the penalties are otherwise identical.
//!
//! Checked: setup (design, penalties, constraint rows, prediction matrix),
//! the fit at each fixture λ (coefficients and binding rows), and the REML
//! optimum (λ, REML, edf, scale, fitted values, Vp, per-term predictions
//! and standard errors), then monotonicity on a dense grid inside the data
//! range. scasm's Fellner–Schall iterations are not compared: gamrs reaches
//! the same optimum by Newton.

use std::path::PathBuf;

use ndarray::{Array1, Array2, Axis};
use serde_json::Value;

use gamrs::design::{Additive, DesignStrategy, MarginKind, Predictor, Shape, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit::shape_constrained::{fit_at_lambda, reml_at_lambda};
use gamrs::fit_with_design;

fn load(name: &str) -> Value {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

/// A number, or a one-element row R's JSON writer left boxed.
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

/// Largest `|a − b| / (1 + |b|)`.
fn max_err(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0, f64::max)
}

fn max_abs(a: impl IntoIterator<Item = f64>) -> f64 {
    a.into_iter().fold(0.0, |m, v| m.max(v.abs()))
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn sc(col: usize) -> TermSpec {
    TermSpec::Sc {
        col,
        k: 10,
        shape: Shape::Decreasing,
    }
}

fn cr(col: usize, by: Option<usize>) -> TermSpec {
    TermSpec::Cr {
        col,
        k: 8,
        pc: None,
        by,
    }
}

fn te_sc(col_a: usize, col_b: usize) -> TermSpec {
    TermSpec::Tensor {
        col_a,
        col_b,
        k_a: 6,
        k_b: 5,
        bs_a: MarginKind::Sc(Some(Shape::Decreasing)),
        bs_b: MarginKind::Sc(Some(Shape::Increasing)),
        by: None,
    }
}

/// Term specs per fixture model; columns follow the fixture's
/// `inputs.columns`.
fn specs(model: &str) -> Vec<TermSpec> {
    match model {
        "mono" => vec![sc(0)],
        "te" => vec![te_sc(0, 1)],
        "sc_cr" => vec![sc(0), cr(1, None)],
        "sc_cr_re" => vec![sc(0), cr(1, None), TermSpec::Re { col: 2, by: None }],
        "tesc_cr" => vec![te_sc(0, 1), cr(2, None)],
        "sc_tecr" => vec![
            sc(0),
            TermSpec::Tensor {
                col_a: 1,
                col_b: 2,
                k_a: 5,
                k_b: 5,
                bs_a: MarginKind::Cr,
                bs_b: MarginKind::Cr,
                by: None,
            },
        ],
        "sc_crby" => vec![sc(0), cr(1, Some(2))],
        _ => unreachable!(),
    }
}

/// (term, column, direction) of each monotone direction to check on a grid.
fn monotone_directions(model: &str) -> Vec<(usize, usize, f64)> {
    match model {
        "te" | "tesc_cr" => vec![(0, 0, -1.0), (0, 1, 1.0)],
        _ => vec![(0, 0, -1.0)],
    }
}

/// Each term's centring map (`β_raw = Z_j β_j`).
fn term_map(p: &Predictor) -> Array2<f64> {
    match p {
        Predictor::Sc(s) => s.centring.clone(),
        Predictor::Cr(c) => c.centring.clone(),
        Predictor::Tensor(t) => t.centring.clone(),
        Predictor::Re(r) => Array2::eye(r.levels.len()),
        Predictor::By(b) => term_map(&b.term),
        _ => unreachable!("term kind not in these fixtures"),
    }
}

/// Block-diagonal `Z` over all coefficients (intercept first).
fn full_map(pred: &Predictor) -> Array2<f64> {
    let Predictor::Additive(ap) = pred else {
        unreachable!()
    };
    let maps: Vec<Array2<f64>> = ap.term_predictors.iter().map(term_map).collect();
    let rows = 1 + maps.iter().map(|m| m.nrows()).sum::<usize>();
    let cols = 1 + maps.iter().map(|m| m.ncols()).sum::<usize>();
    let mut z = Array2::<f64>::zeros((rows, cols));
    z[[0, 0]] = 1.0;
    let (mut r, mut c) = (1, 1);
    for m in &maps {
        z.slice_mut(ndarray::s![r..r + m.nrows(), c..c + m.ncols()])
            .assign(m);
        r += m.nrows();
        c += m.ncols();
    }
    z
}

fn check(name: &str, model: &str) {
    let fx = load(name);
    // The single-term fixtures point at their source fixture for the data.
    let data = match fx["inputs"]["source_fixture"].as_str() {
        Some(src) if fx["inputs"]["x_train"].is_null() => load(src.trim_end_matches(".json")),
        _ => fx.clone(),
    };
    let x = mat_of(&data["inputs"]["x_train"]);
    let y = Array1::from_vec(vec_of(&data["inputs"]["y_train"]));
    let x_new = mat_of(&fx["inputs"]["x_new"]);
    let terms = specs(model);
    let prep = Additive {
        terms: terms.clone(),
    }
    .prepare(x.view())
    .unwrap();
    let z = full_map(&prep.predictor);
    let setup = &fx["setup"];
    let tag = name.trim_start_matches("scasm_internals_");

    // ── Setup ────────────────────────────────────────────────────────────
    let head = mat_of(&setup["X_head"]);
    let mine_head = prep
        .x_design
        .slice(ndarray::s![0..head.nrows(), ..])
        .to_owned();
    let e_x = max_err(&mine_head, &head.dot(&z));
    // Penalties: Zᵀ S_scasm Z = c_j · S_gamrs, c_j = 1 except for te terms.
    let mut first_of = Vec::new();
    for s in setup["smooths"].as_array().unwrap() {
        let n_pen = s["S_scale"].as_array().map_or(1, |a| a.len());
        for _ in 0..n_pen {
            first_of.push(s["first_para"].as_u64().unwrap() as usize - 1);
        }
    }
    let s_ref: Vec<Array2<f64>> = setup["S"].as_array().unwrap().iter().map(mat_of).collect();
    assert_eq!(s_ref.len(), prep.s_list.len(), "{tag}: number of penalties");
    let mut lambda_factor = Vec::new();
    let mut e_s = 0.0_f64;
    for (j, s) in s_ref.iter().enumerate() {
        let mut full = Array2::<f64>::zeros((z.nrows(), z.nrows()));
        let f = first_of[j];
        full.slice_mut(ndarray::s![f..f + s.nrows(), f..f + s.ncols()])
            .assign(s);
        let projected = z.t().dot(&full).dot(&z);
        let mine = &prep.s_list[j];
        let norm = |m: &Array2<f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
        let c = norm(mine) / norm(&projected);
        let smax = max_abs(mine.iter().copied());
        e_s = e_s.max(max_abs((&(&projected * c) - mine).iter().copied()) / smax);
        // λ_gamrs S_gamrs = λ_mgcv S_mgcv.
        lambda_factor.push(1.0 / c);
    }
    let ain = mat_of(&setup["Ain"]);
    let mine_rows = &prep.constraints.as_ref().unwrap().rows;
    let e_a = max_err(mine_rows, &ain.dot(&z));
    let lp_new = mat_of(&setup["lpmatrix_new"]);
    let e_lp = max_err(
        &prep.predictor.design(x_new.view()).unwrap(),
        &lp_new.dot(&z),
    );
    eprintln!(
        "{tag}: setup X {e_x:.1e}, S {e_s:.1e}, Ain {e_a:.1e}, prediction matrix {e_lp:.1e}; \
         λ factors {lambda_factor:.4?}"
    );
    assert!(
        e_x < 1e-12 && e_s < 1e-12 && e_a < 1e-12 && e_lp < 1e-12,
        "{tag}: setup off"
    );

    let prepare = || {
        Additive {
            terms: terms.clone(),
        }
        .prepare(x.view())
        .unwrap()
    };
    let to_gamrs =
        |sp: &[f64]| -> Array1<f64> { sp.iter().zip(&lambda_factor).map(|(s, f)| s * f).collect() };

    // ── Fixed λ: coefficients and binding rows ───────────────────────────
    let mut e_fixed = 0.0_f64;
    for rung in fx["fixed_sp"].as_array().unwrap() {
        let sp = vec_of(&rung["sp"]);
        let (fit, active) = fit_at_lambda(prepare(), y.view(), None, to_gamrs(&sp).view()).unwrap();
        let b = z.dot(&fit.beta);
        let b_ref = vec_of(&rung["coefficients"]);
        e_fixed = e_fixed.max(max_abs(
            b.iter().zip(&b_ref).map(|(a, r)| (a - r) / (1.0 + r.abs())),
        ));
        let act: Vec<usize> = active.iter().map(|a| a + 1).collect();
        assert_eq!(
            act,
            ints_of(&rung["active"]),
            "{tag}: binding rows at sp {sp:?}"
        );
    }
    eprintln!("{tag}: fixed-λ coefficients {e_fixed:.1e}");
    assert!(e_fixed < 2e-9, "{tag}: fixed-λ coefficients off");

    // ── Everything scasm reports, at scasm's final λ ─────────────────────
    let fin = &fx["final"];
    let sp_ref = to_gamrs(&vec_of(&fin["sp"]));
    let (at_ref, _) = fit_at_lambda(prepare(), y.view(), None, sp_ref.view()).unwrap();
    let mut e = reported(&at_ref, fin, &x, &x_new, &z);
    // scasm reports the LAML with its scale `rss/(n − edf)`; gamrs's REML
    // value is the profiled criterion it minimises. Check scasm's number
    // from gamrs's pieces.
    e[0] = rel(
        scasm_reml(&prep, &y, &at_ref, &sp_ref),
        fin["reml"].as_f64().unwrap(),
    );
    assert!(
        rel(
            at_ref.reml_value,
            reml_at_lambda(&prep, y.view(), None, sp_ref.view()).unwrap()
        ) < 1e-12,
        "{tag}: reported REML is not the criterion"
    );
    eprintln!(
        "{tag}: at scasm's λ: REML {:.1e}, edf {:.1e}, scale {:.1e}, fitted {:.1e}, Vp {:.1e}, \
         terms {:.1e}, se {:.1e}",
        e[0], e[1], e[2], e[3], e[4], e[5], e[6]
    );
    assert!(
        e[0] < 1e-12 && e[1] < 1e-9 && e[2] < 1e-11 && e[3] < 1e-9,
        "{tag}: fit at scasm's λ off"
    );
    assert!(
        e[4] < 1e-11 && e[5] < 1e-9 && e[6] < 1e-9,
        "{tag}: inference at scasm's λ off"
    );

    // ── The REML optimum ─────────────────────────────────────────────────
    // gamrs minimises the REML of the centred model at the constrained fit
    // (the profiled-scale form of the LAML scasm reports); scasm stops its
    // Fellner–Schall iteration on its own reduced criterion once penalties
    // move by under 1%. On gamrs's criterion its optimum must be at least
    // as good as scasm's λ.
    let fit = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: terms.clone(),
        },
        x.view(),
        y.view(),
        None,
    )
    .unwrap();
    assert!(fit.converged, "{tag}: not converged");
    let crit_ref = reml_at_lambda(&prep, y.view(), None, sp_ref.view()).unwrap();
    let e = reported(&fit, fin, &x, &x_new, &z);
    let d_log_sp = max_abs(fit.lambda.iter().zip(&sp_ref).map(|(a, b)| a.ln() - b.ln()));
    let sd_y = y.std(0.0);
    eprintln!(
        "{tag}: optimum ({} Newton iterations): criterion gamrs − scasm {:.2e}; vs scasm \
         log λ {d_log_sp:.1e}, edf {:.1e}, fitted {:.1e} ({:.1e} of sd y), terms {:.1e}, \
         se {:.1e}",
        fit.n_iters,
        fit.reml_value - crit_ref,
        e[1],
        e[3],
        e[3] / sd_y,
        e[5],
        e[6],
    );
    assert!(
        fit.reml_value <= crit_ref + 1e-9 * crit_ref.abs(),
        "{tag}: gamrs optimum worse than scasm's"
    );
    assert!(e[3] / sd_y < 5e-2, "{tag}: fitted values far from scasm's");

    monotone_on_grid(&fit, &x, &monotone_directions(model), tag);
}

/// scasm's LAML at `fit`: `(−Dp/φ + log|S_λ|₊ − log|XᵀX + S_λ|)/2 +
/// M log(2π)/2 − n log(2πφ)/2` with φ the reported scale, returned as
/// −LAML.
fn scasm_reml(
    prep: &gamrs::design::PreparedDesign,
    y: &Array1<f64>,
    fit: &gamrs::FittedGam,
    lambda: &Array1<f64>,
) -> f64 {
    use ndarray_linalg::{Cholesky, Eigh, UPLO};
    let x: Array2<f64> = prep.x_design.view().to_owned();
    let p = x.ncols();
    let mut s = Array2::<f64>::zeros((p, p));
    for (j, sj) in prep.s_list.iter().enumerate() {
        s.scaled_add(lambda[j], sj);
    }
    let (ev, _) = s.eigh(UPLO::Lower).unwrap();
    let tol = ev.iter().fold(0.0_f64, |m, v| m.max(*v)) * 1e-10;
    let pos: Vec<f64> = ev.iter().copied().filter(|v| *v > tol).collect();
    let log_det_s: f64 = pos.iter().map(|v| v.ln()).sum();
    let a = x.t().dot(&x) + &s;
    let l = a.cholesky(UPLO::Lower).unwrap();
    let log_det_a: f64 = 2.0 * l.diag().iter().map(|v| v.ln()).sum::<f64>();
    let r = y - &x.dot(&fit.beta);
    let dp = r.dot(&r) + fit.beta.dot(&s.dot(&fit.beta));
    let (n, phi) = (y.len() as f64, fit.scale);
    let two_pi = 2.0 * std::f64::consts::PI;
    let laml = (-dp / phi + log_det_s - log_det_a) / 2.0
        + (p - pos.len()) as f64 * two_pi.ln() / 2.0
        - n * (two_pi * phi).ln() / 2.0;
    -laml
}

/// Relative REML, absolute edf, relative scale, absolute fitted,
/// `max_err` Vp, absolute term predictions and relative term standard
/// errors of `fit` against scasm's `final` block.
fn reported(
    fit: &gamrs::FittedGam,
    fin: &Value,
    x: &Array2<f64>,
    x_new: &Array2<f64>,
    z: &Array2<f64>,
) -> [f64; 7] {
    let e_reml = rel(fit.reml_value, fin["reml"].as_f64().unwrap());
    let e_edf = (fit.edf_total - fin["edf_total"].as_f64().unwrap()).abs();
    let e_scale = rel(fit.scale, fin["scale"].as_f64().unwrap());
    let e_fit = max_abs(
        fit.predict(x.view())
            .unwrap()
            .iter()
            .zip(vec_of(&fin["fitted"]))
            .map(|(a, b)| a - b),
    );
    let vp = z.dot(&fit.vcov).dot(&z.t());
    let e_vp = max_err(&vp, &mat_of(&fin["Vp"]));
    let Predictor::Additive(ap) = &fit.predictor else {
        unreachable!()
    };
    let lp = fit.predictor.design(x_new.view()).unwrap();
    let (mut e_t, mut e_se) = (0.0_f64, 0.0_f64);
    for (j, &(a, b)) in ap.term_col_ranges.iter().enumerate() {
        let xj = lp.slice(ndarray::s![.., a..b]);
        let t = xj.dot(&fit.beta.slice(ndarray::s![a..b]));
        let v = fit.vcov.slice(ndarray::s![a..b, a..b]);
        let se = (&xj.dot(&v) * &xj).sum_axis(Axis(1)).mapv(f64::sqrt);
        e_t = e_t.max(max_abs(
            t.iter()
                .zip(vec_of(&fin["term_new"][j]))
                .map(|(a, b)| a - b),
        ));
        e_se = e_se.max(
            se.iter()
                .zip(vec_of(&fin["term_new_se"][j]))
                .map(|(a, b)| rel(*a, b))
                .fold(0.0, f64::max),
        );
    }
    [e_reml, e_edf, e_scale, e_fit, e_vp, e_t, e_se]
}

/// Evaluate the term along `col` on a 60-point grid spanning the data, with
/// the other margin (for a tensor) on 15 values across its data range and
/// every other column at its mid-range, and assert each step moves in
/// direction `dir`. The grid stays inside the data rectangle: a tensor is
/// monotone only there.
fn monotone_on_grid(
    fit: &gamrs::FittedGam,
    x: &Array2<f64>,
    checks: &[(usize, usize, f64)],
    tag: &str,
) {
    let Predictor::Additive(ap) = &fit.predictor else {
        unreachable!()
    };
    let lo = x.fold_axis(Axis(0), f64::INFINITY, |a, b| a.min(*b));
    let hi = x.fold_axis(Axis(0), f64::NEG_INFINITY, |a, b| a.max(*b));
    for &(term, col, dir) in checks {
        let (a, b) = ap.term_col_ranges[term];
        let other = ap.cols_used[term].iter().copied().find(|&c| c != col);
        let n_other = if other.is_some() { 15 } else { 1 };
        let mut worst = f64::INFINITY;
        for o in 0..n_other {
            let mut g = Array2::<f64>::zeros((60, x.ncols()));
            for c in 0..x.ncols() {
                g.column_mut(c).fill((lo[c] + hi[c]) / 2.0);
            }
            for i in 0..60 {
                g[[i, col]] = lo[col] + (hi[col] - lo[col]) * i as f64 / 59.0;
            }
            if let Some(oc) = other {
                let v = lo[oc] + (hi[oc] - lo[oc]) * o as f64 / (n_other - 1) as f64;
                g.column_mut(oc).fill(v);
            }
            let d = fit.predictor.design(g.view()).unwrap();
            let f = d
                .slice(ndarray::s![.., a..b])
                .dot(&fit.beta.slice(ndarray::s![a..b]));
            for i in 1..60 {
                worst = worst.min(dir * (f[i] - f[i - 1]));
            }
        }
        assert!(
            worst > -1e-10,
            "{tag}: term {term} not monotone along column {col} (worst step {worst:.2e})"
        );
    }
}

macro_rules! cases {
    ($($test:ident: $name:expr, $model:expr;)*) => {$(
        #[test]
        fn $test() {
            check($name, $model);
        }
    )*};
}

cases! {
    mono_n500: "scasm_internals_mono_n500_k10", "mono";
    mono_n3000: "scasm_internals_mono_n3000_k10", "mono";
    te_n500: "scasm_internals_te_n500_k6x5", "te";
    te_n3000: "scasm_internals_te_n3000_k6x5", "te";
    sc_cr_n500: "scasm_internals_sc_cr_n500", "sc_cr";
    sc_cr_n3000: "scasm_internals_sc_cr_n3000", "sc_cr";
    sc_cr_re_n500: "scasm_internals_sc_cr_re_n500", "sc_cr_re";
    sc_cr_re_n3000: "scasm_internals_sc_cr_re_n3000", "sc_cr_re";
    tesc_cr_n500: "scasm_internals_tesc_cr_n500", "tesc_cr";
    tesc_cr_n3000: "scasm_internals_tesc_cr_n3000", "tesc_cr";
    sc_tecr_n500: "scasm_internals_sc_tecr_n500", "sc_tecr";
    sc_tecr_n3000: "scasm_internals_sc_tecr_n3000", "sc_tecr";
    sc_crby_n500: "scasm_internals_sc_crby_n500", "sc_crby";
    sc_crby_n3000: "scasm_internals_sc_crby_n3000", "sc_crby";
}

/// A non-Gaussian family with an sc term is refused with a clear error.
#[test]
fn sc_term_rejected_for_non_gaussian() {
    let n = 200;
    let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    let y = Array1::from_shape_fn(n, |i| (i % 3 == 0) as u8 as f64);
    let err = fit_with_design(
        gamrs::family::bernoulli_logit(),
        Additive { terms: vec![sc(0)] },
        x.view(),
        y.view(),
        None,
    )
    .err()
    .expect("bernoulli with an sc term must fail");
    let msg = err.to_string();
    assert!(msg.contains("Gaussian"), "unhelpful error: {msg}");
}
