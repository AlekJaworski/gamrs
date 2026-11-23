//! mgcv's `gam(select = TRUE)`: one extra penalty per smooth on the directions
//! its own penalties leave free, so REML can remove a whole term.
//!
//! Parity against mgcv (`scripts/r/gen_select_fixtures.R`) on three fits:
//! CR smooths with one noise term, `te(x, z)`, and `te(x, z, by = w) + s(v)`
//! with a tensor that carries no signal, which `select` must shrink to zero.

use std::path::PathBuf;

use ndarray::{s, Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, MarginKind, Predictor, Select, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::{fit_with_design, DesignStrategy, FittedGam};

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
    predictions_new: Vec<f64>,
    edf_per_term: Vec<f64>,
    reml: f64,
    scale: f64,
    sp: Vec<f64>,
}

const CR: &str = "select_cr_gaussian_n400";
const TE: &str = "select_te_gaussian_n400_k4x3";
const TE_BY_NULL: &str = "select_te_by_null_gaussian_n400_k4x3";

fn load(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p).expect("missing fixture");
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn rows(v: &[Vec<f64>]) -> Array2<f64> {
    Array2::from_shape_vec((v.len(), v[0].len()), v.concat()).unwrap()
}

fn cr(col: usize, k: usize) -> TermSpec {
    TermSpec::Cr {
        col,
        k,
        pc: None,
        by: None,
    }
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

fn terms(name: &str) -> Vec<TermSpec> {
    match name {
        CR => vec![cr(0, 10), cr(1, 8), cr(2, 8)],
        TE => vec![te(None)],
        TE_BY_NULL => vec![te(Some(2)), cr(3, 8)],
        _ => unreachable!(),
    }
}

fn term_contribution(fit: &FittedGam, x: &Array2<f64>, j: usize) -> Array1<f64> {
    let Predictor::Additive(additive) = &fit.predictor else {
        panic!("expected an additive predictor");
    };
    let (a, b) = additive.term_col_ranges[j];
    let design = fit.predictor.design(x.view()).unwrap();
    design.slice(s![.., a..b]).dot(&fit.beta.slice(s![a..b]))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

fn fit(name: &str) -> (FittedGam, Fixture) {
    let fx = load(name);
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let f = fit_with_design(
        gaussian_identity(),
        Select(Additive { terms: terms(name) }),
        x.view(),
        y.view(),
        None,
    )
    .expect("fit failed");
    (f, fx)
}

fn assert_parity(name: &str) {
    let (f, fx) = fit(name);
    let m = &fx.mgcv_output;
    let x = rows(&fx.inputs.x_train);
    let x_new = rows(&fx.inputs.x_new);
    assert!(f.converged);
    assert_eq!(f.lambda.len(), m.sp.len(), "one λ per mgcv sp");

    let d_fitted = max_abs_diff(f.predict(x.view()).unwrap().as_slice().unwrap(), &m.fitted);
    let d_new = max_abs_diff(
        f.predict(x_new.view()).unwrap().as_slice().unwrap(),
        &m.predictions_new,
    );
    let d_terms = (0..m.term_train.len())
        .map(|j| {
            max_abs_diff(
                term_contribution(&f, &x, j).as_slice().unwrap(),
                &m.term_train[j],
            )
        })
        .fold(0.0, f64::max);
    let d_edf = max_abs_diff(f.edf_per_term.as_slice().unwrap(), &m.edf_per_term);
    let d_reml = (f.reml_value - m.reml).abs();
    let d_scale = (f.scale - m.scale).abs() / m.scale;
    println!(
        "[{name}] fitted {d_fitted:.2e}  new {d_new:.2e}  terms {d_terms:.2e}  edf {d_edf:.2e} \
         ({:?} vs {:?})  reml {d_reml:.2e}  scale rel {d_scale:.2e}\n  λ {:?}\n  sp {:?}",
        f.edf_per_term.to_vec(),
        m.edf_per_term,
        f.lambda.to_vec(),
        m.sp
    );
    // A shrunk term's λ runs to infinity along a ridge where REML barely moves.
    // The fixtures converge mgcv far past its defaults; gamrs stops by mgcv's
    // default rules, 1e-5 to 1.3e-4 above that optimum, where mgcv's own
    // defaults stop 6e-5 to 4.4e-4 above it.
    assert!(d_fitted < 1e-4, "fitted values off mgcv by {d_fitted:.3e}");
    assert!(d_new < 1e-4, "new-data predictions off mgcv by {d_new:.3e}");
    assert!(
        d_terms < 1e-4,
        "term contributions off mgcv by {d_terms:.3e}"
    );
    assert!(d_edf < 1e-3, "edf off mgcv by {d_edf:.3e}");
    assert!(d_reml < 2e-4, "REML off mgcv by {d_reml:.3e}");
    assert!(d_scale < 1e-5, "scale off mgcv by {d_scale:.3e}");
}

#[test]
fn select_cr_matches_mgcv() {
    assert_parity(CR);
}

#[test]
fn select_te_matches_mgcv() {
    assert_parity(TE);
}

#[test]
fn select_te_by_matches_mgcv() {
    assert_parity(TE_BY_NULL);
}

/// Without `select` the tensor keeps the 4 directions its margins leave free
/// (edf ≥ 4); with it the term goes to ~0 and contributes ~nothing.
#[test]
fn select_removes_a_term_with_no_signal() {
    let fx = load(TE_BY_NULL);
    let x = rows(&fx.inputs.x_train);
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let plain = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: terms(TE_BY_NULL),
        },
        x.view(),
        y.view(),
        None,
    )
    .unwrap();
    let (selected, _) = fit(TE_BY_NULL);
    assert!(
        plain.edf_per_term[0] > 3.99,
        "edf {}",
        plain.edf_per_term[0]
    );
    assert!(
        selected.edf_per_term[0] < 0.01,
        "edf {}",
        selected.edf_per_term[0]
    );
    let term = term_contribution(&selected, &x, 0);
    let sd_y = y.std(1.0);
    assert!(term.iter().all(|v| v.abs() < 1e-3 * sd_y));
}

/// The null-space penalty sits after each term's own penalties, spans exactly
/// what they leave free, and leaves a full-rank random effect alone.
#[test]
fn null_space_penalties_follow_mgcv_layout() {
    let fx = load(TE_BY_NULL);
    let x = rows(&fx.inputs.x_train);
    let plain = Additive {
        terms: terms(TE_BY_NULL),
    }
    .prepare(x.view())
    .unwrap();
    let selected = Select(Additive {
        terms: terms(TE_BY_NULL),
    })
    .prepare(x.view())
    .unwrap();
    assert_eq!(plain.s_list.len(), 3);
    assert_eq!(selected.s_list.len(), 5);
    assert_eq!(selected.s_list[0], plain.s_list[0]);
    assert_eq!(selected.s_list[1], plain.s_list[1]);
    assert_eq!(selected.s_list[3], plain.s_list[2]);
    // Uncentred by-tensor: 2 × 2 free directions; centred CR: 1.
    assert_eq!(selected.rank_s_list[2], 4);
    assert_eq!(selected.rank_s_list[4], 1);
    assert_eq!(plain.mp, 1 + 4 + 1);
    assert_eq!(selected.mp, 1);

    let re = Select(Additive {
        terms: vec![TermSpec::Re { col: 0, by: None }],
    })
    .prepare(Array2::from_shape_fn((12, 1), |(i, _)| (i % 4) as f64).view())
    .unwrap();
    assert_eq!(re.s_list.len(), 1);
}
