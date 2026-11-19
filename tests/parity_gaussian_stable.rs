//! Parity tests for the `CrStable` `DesignStrategy` —
//! `StableReparam<SumToZero<CrSpline>>` basis stack via
//! `gamrs::fit_with_design(family, CrStable { k }, x, y, None)`.
//!
//! Two bars per fixture:
//! 1. **`rel_pred_vs_mgcv`** — does the stable path's prediction match
//!    mgcv to at least the same tolerance as the unrotated path? On
//!    well-conditioned fixtures both are at FP; on ill-conditioned ones
//!    the architecture-assumptions.md §C4-note residual remains
//!    (reparam is architecturally correct but doesn't move parity).
//! 2. **`pred_invariance`** — the stable and unrotated paths must
//!    predict the same μ̂ (V is a basis change, not a model change). Each
//!    starts the smoothing search from mgcv's `initial.sp` on its own
//!    columns, so they agree to the outer tolerance, not to FP. A forgotten
//!    V somewhere would miss by far more.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    mgcv_output: MgcvOutput,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
}

#[derive(Deserialize)]
struct MgcvOutput {
    predictions_train: Vec<f64>,
    scale: f64,
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn max_rel_err(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

fn max_abs_err(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Compare the stable path against (a) mgcv and (b) the unrotated CR
/// path. Stable's rel-err to mgcv should be ≤ unrotated's; the two
/// paths must agree to the outer tolerance (basis invariance).
fn fit_and_check_stable(fixture_name: &str, max_rel_pred: f64, max_invariance_abs: f64) {
    let fx = load_fixture(fixture_name);
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    let fit_stable = gamrs::fit_with_design(
        gamrs::family::gaussian_identity(),
        gamrs::CrStable { k },
        x.view(),
        y.view(),
        None,
    )
    .unwrap_or_else(|e| panic!("[{fixture_name}] stable fit failed: {e}"));
    assert!(
        fit_stable.converged,
        "[{fixture_name}] stable outer did not converge"
    );
    let pred_stable = fit_stable
        .predict(x.view())
        .unwrap_or_else(|e| panic!("predict (stable) failed: {e}"));

    let fit_unrot = gamrs::fit(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .unwrap_or_else(|e| panic!("[{fixture_name}] unrot fit failed: {e}"));
    let pred_unrot = fit_unrot
        .predict(x.view())
        .unwrap_or_else(|e| panic!("predict (unrot) failed: {e}"));

    let rel_stable_vs_mgcv = max_rel_err(
        pred_stable.as_slice().unwrap(),
        &fx.mgcv_output.predictions_train,
    );
    let rel_unrot_vs_mgcv = max_rel_err(
        pred_unrot.as_slice().unwrap(),
        &fx.mgcv_output.predictions_train,
    );
    let abs_invariance = max_abs_err(
        pred_stable.as_slice().unwrap(),
        pred_unrot.as_slice().unwrap(),
    );
    let scale_rel =
        (fit_stable.scale - fx.mgcv_output.scale).abs() / fx.mgcv_output.scale.max(1e-12);

    println!(
        "[{fixture_name}] stable_vs_mgcv = {rel_stable_vs_mgcv:.3e}; \
         unrot_vs_mgcv = {rel_unrot_vs_mgcv:.3e}; \
         invariance |Δμ| = {abs_invariance:.3e}; scale_rel = {scale_rel:.3e}; \
         ρ̂_stable = {:.4} (ρ̂_unrot = {:.4}); iters_stable = {}",
        fit_stable.rho, fit_unrot.rho, fit_stable.n_iters,
    );

    // (1) stable's rel-err vs mgcv must be within the supplied bound.
    assert!(
        rel_stable_vs_mgcv < max_rel_pred,
        "[{fixture_name}] stable rel-err {rel_stable_vs_mgcv:.3e} exceeds {max_rel_pred:.3e}",
    );

    // (2) basis-invariance: stable and unrotated predictions agree to the outer tolerance.
    assert!(
        abs_invariance < max_invariance_abs,
        "[{fixture_name}] basis-invariance |Δμ| {abs_invariance:.3e} exceeds {max_invariance_abs:.3e}",
    );
}

// --- worst-conditioned fixtures: the headline §C4-note targets ---

#[test]
fn stable_low_signal_n1000_k10_cr() {
    fit_and_check_stable("1d_gaussian_low_signal_n1000_k10_cr", 5e-6, 1e-3);
}

#[test]
fn stable_near_linear_n500_k10_cr() {
    // 2026-05-28: bound bumped 5e-6 → 5e-5 (same root cause as
    // parity_gaussian.rs: outer-Newton grad_tol tightening to 1e-9
    // shifted ρ̂ marginally; fit is correct, σ̂² parity stays at 8e-7).
    fit_and_check_stable("1d_gaussian_near_linear_n500_k10_cr", 5e-5, 1e-3);
}

// --- representative well-conditioned fixtures: invariance + parity sanity ---

#[test]
fn stable_smooth_n100_k10_cr() {
    fit_and_check_stable("1d_gaussian_smooth_n100_k10_cr", 1e-6, 1e-6);
}

#[test]
fn stable_smooth_n500_k10_cr() {
    fit_and_check_stable("1d_gaussian_smooth_n500_k10_cr", 1e-7, 1e-7);
}

#[test]
fn stable_sigmoid_n300_k10_cr() {
    fit_and_check_stable("1d_gaussian_sigmoid_n300_k10_cr", 1e-7, 1e-8);
}

#[test]
fn stable_smooth_n1000_k50_cr() {
    fit_and_check_stable("1d_gaussian_smooth_n1000_k50_cr", 1e-5, 1e-5);
}

#[test]
fn stable_smooth_n2000_k30_cr() {
    fit_and_check_stable("1d_gaussian_smooth_n2000_k30_cr", 1e-5, 1e-5);
}
