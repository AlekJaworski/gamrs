//! The Gaussian REML search takes mgcv's path, so it lands in the same local
//! minimum.
//!
//! REML over several smooths is not always unimodal: on pricing-style designs
//! with raw-scale covariates (dollars, square feet) and terms near a straight
//! line, it can have more than one local minimum. Which one a search reaches
//! depends on its whole path, and matching mgcv took all of:
//! - mgcv's start, `initial.spg` (for a Gaussian, `initial.sp` on `√w·X`).
//!   From λ = 1, mgcv itself stops in a higher minimum on
//!   `gaussian_start_basin_n300`; gamrs started from ρ = 0.
//! - mgcv's parameters: ρ and log φ jointly, from `log(null.scale/10)`
//!   (`score::joint_scale`), not ρ with φ profiled out. On
//!   `gaussian_start_basin_n300`, mgcv's start alone still leaves the
//!   profiled search in the higher basin.
//! - mgcv's handling of an indefinite Hessian
//!   (`NewtonOpts::mgcv_indefinite`). Without it, the search on
//!   `gaussian_indefinite_step_n320` reaches a minimum 0.099 BELOW mgcv's.
//!   These tests pin gamrs to mgcv's minimum, not to the lowest one.
//!
//! The endpoint checks catch a changed basin; the path checks catch the drift
//! that precedes one. Both fixtures are synthetic
//! (`scripts/r/gen_search_path_fixtures.R`), stand-ins for KBB pricing fits
//! where these differences were first seen.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, DesignStrategy, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit::{LambdaInit, MgcvInit};
use gamrs::outer::record_outer_path;
use gamrs::{fit_with_design, FittedGam};

#[derive(Deserialize)]
struct Fixture {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    /// Basis size per column; 0 is a linear (parametric) term.
    k: Vec<usize>,
    initial_log_sp: Vec<f64>,
    strict: Strict,
    /// mgcv's `(log sp, log scale)` at the start of each `newton()` iteration.
    mgcv_path: Vec<Vec<f64>>,
    from_sp1: Option<OtherMinimum>,
    lower_minimum: Option<OtherMinimum>,
}

#[derive(Deserialize)]
struct Strict {
    reml: f64,
    sp: Vec<f64>,
    edf_total: f64,
    fitted: Vec<f64>,
}

#[derive(Deserialize)]
struct OtherMinimum {
    reml: f64,
}

fn load(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(format!("tests/fixtures/strict/{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn setup(f: &Fixture) -> (Array2<f64>, Array1<f64>, Additive) {
    let d = f.x_train[0].len();
    let flat: Vec<f64> = f.x_train.iter().flatten().copied().collect();
    let x = Array2::from_shape_vec((f.x_train.len(), d), flat).unwrap();
    let y = Array1::from_vec(f.y_train.clone());
    let terms =
        f.k.iter()
            .enumerate()
            .map(|(col, &k)| match k {
                0 => TermSpec::Parametric { col },
                k => TermSpec::Cr {
                    col,
                    k,
                    pc: None,
                    by: None,
                },
            })
            .collect();
    (x, y, Additive { terms })
}

/// The fit and the outer search's iterates.
fn fit(f: &Fixture) -> (Array2<f64>, FittedGam, Vec<Vec<f64>>) {
    let (x, y, design) = setup(f);
    let (fit, path) = record_outer_path(|| {
        fit_with_design(gaussian_identity(), design, x.view(), y.view(), None).unwrap()
    });
    assert!(fit.converged);
    (x, fit, path)
}

fn check_start(name: &str) {
    let f = load(name);
    let (x, y, design) = setup(&f);
    let prep = design.prepare(x.view()).unwrap();
    let rho0 = MgcvInit::default().init(y.view(), &prep.x_design, &prep.s_list);
    let d = rho0
        .iter()
        .zip(&f.initial_log_sp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        d < 1e-8,
        "[{name}] start off mgcv's initial.spg by {d:.3e}: {rho0}"
    );
}

fn check_minimum(name: &str) {
    let f = load(name);
    let (x, fit, _) = fit(&f);
    let d_reml = fit.reml_value - f.strict.reml;
    let fitted = fit.predict(x.view()).unwrap();
    let d_fit = fitted
        .iter()
        .zip(&f.strict.fitted)
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max);
    let d_edf = (fit.edf_total - f.strict.edf_total).abs();
    println!(
        "[path] {name} reml(gamrs-mgcv) {d_reml:+.2e} fitted {d_fit:.1e} edf {d_edf:.1e} \
         log λ {:?} (mgcv {:?})",
        fit.lambda.mapv(f64::ln).to_vec(),
        f.strict.sp.iter().map(|s| s.ln()).collect::<Vec<_>>()
    );
    assert!(
        (-1e-8..1e-6).contains(&d_reml),
        "[{name}] REML − mgcv's minimum = {d_reml:+.3e}"
    );
    assert!(
        d_fit < 1e-7,
        "[{name}] fitted values off mgcv by {d_fit:.3e}"
    );
    assert!(d_edf < 1e-5, "[{name}] EDF off mgcv by {d_edf:.3e}");
}

/// gamrs's first `n` outer iterates against mgcv's, `(ρ, log φ)` each.
fn check_path(name: &str, n: usize, tol: f64) {
    let f = load(name);
    let (_, _, path) = fit(&f);
    let n = n.min(path.len());
    assert!(
        path.len() >= n && f.mgcv_path.len() >= n,
        "[{name}] path too short: gamrs {}, mgcv {}",
        path.len(),
        f.mgcv_path.len()
    );
    for (i, (a, b)) in path.iter().zip(&f.mgcv_path).take(n).enumerate() {
        assert_eq!(a.len(), b.len(), "[{name}] iterate {i}: dimension");
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        println!("[path] {name} iterate {i}: max |Δ| {d:.2e}");
        assert!(
            d < tol,
            "[{name}] iterate {i} off mgcv's by {d:.3e}\n  gamrs {a:.4?}\n  mgcv  {b:.4?}"
        );
    }
}

const START_BASIN: &str = "gaussian_start_basin_n300";
const INDEFINITE: &str = "gaussian_indefinite_step_n320";

#[test]
fn fixtures_have_other_minima() {
    let f = load(START_BASIN);
    let higher = f.from_sp1.expect("from_sp1").reml;
    assert!(
        higher - f.strict.reml > 0.5,
        "start-basin fixture no longer separates the minima: {higher} vs {}",
        f.strict.reml
    );
    let f = load(INDEFINITE);
    let lower = f.lower_minimum.expect("lower_minimum").reml;
    assert!(
        f.strict.reml - lower > 0.05,
        "indefinite-step fixture lost its lower minimum: {lower} vs {}",
        f.strict.reml
    );
}

#[test]
fn start_matches_mgcv_initial_spg() {
    check_start(START_BASIN);
    check_start(INDEFINITE);
}

#[test]
fn start_basin_lands_in_mgcvs_minimum() {
    check_minimum(START_BASIN);
}

#[test]
fn indefinite_step_lands_in_mgcvs_minimum() {
    check_minimum(INDEFINITE);
}

#[test]
fn start_basin_follows_mgcvs_path() {
    check_path(START_BASIN, PATH_ITERS, PATH_TOL);
}

#[test]
fn indefinite_step_follows_mgcvs_path() {
    check_path(INDEFINITE, PATH_ITERS, PATH_TOL);
}

/// Iterates compared, and the bound on each. Measured when the fixtures were
/// made (2026-11-17): through iterate 18 both paths agree with mgcv's to
/// ≤ 1.2e-7 (≤ 6.1e-9 through 15); after that the axes heading to λ = ∞ are
/// frozen at different iterations, which moves no fitted value. Fifteen
/// iterates is well past the steps that decide the basin.
const PATH_ITERS: usize = 15;
const PATH_TOL: f64 = 1e-6;
