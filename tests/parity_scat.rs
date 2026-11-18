//! Phase 2 parity test: gamrs's `fit_scat_cr` (TDist + identity link +
//! joint outer Newton over `[log λ, log σ², log(ν-2)]`) vs mgcv on the
//! `1d_scat_unweighted_n300_k10_cr` fixture.
//!
//! The bound is 1e-3 on μ relative, and it is a measured lock, not a smoke
//! test. It used to be 5e-2 because `TDist::score_rank_adjustment` returned
//! −1, which subtracted one from the penalty null-space rank inside the
//! score's `log|λS|₊`. That tilts the REML surface by ρ/2 per smooth, so the
//! outer Newton converged to a genuinely under-penalised λ — mgcv's own
//! fixed-sp REML sweep on a real `garage_spaces` term put gamrs' answer
//! ~30× under-penalised (edf 4.02 against mgcv's 2.37). Dropping the
//! adjustment moved every scat fixture ~8-16× closer to mgcv; this fixture
//! went 4.1e-3 → 3.8e-4.

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
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let s: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (s / n).sqrt()
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

#[test]
fn scat_unweighted_n300_k10_cr() {
    let fx = load_fixture("1d_scat_unweighted_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    // Seed Newton with sensible defaults: ν = 5 (close to scat's typical
    // optimum), σ² = var(y) / 10 (rough first-cut).
    let y_var = {
        let mean = y.iter().sum::<f64>() / (y.len() as f64);
        let var = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        var
    };
    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("gamrs::fit (scat/TDist) should not fail");
    assert!(fit.converged, "scat outer did not converge");

    let pred = fit.predict(x.view()).expect("predict should not fail");

    let rmse_v = rmse(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    let rel = max_rel_err(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    let abs_e = max_abs_err(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    println!(
        "[scat unweighted n300] rmse = {rmse_v:.3e}; max_rel = {rel:.3e}; max_abs = {abs_e:.3e}; \
         ρ̂ = {:.3}; σ̂² = {:.4}; iters = {}; edf = {:.2}",
        fit.rho[0], fit.scale, fit.n_iters, fit.edf_total,
    );

    // Bar 1e-6: observed 1.132e-7 (ρ̂ = 4.795, edf 8.79 against mgcv's
    // fixture). ~9x headroom. Was a 1e-3 bar against a 3.8e-4 observation;
    // the observed-curvature criterion took this fixture down by 3400x, and a
    // bar left at 1e-3 would no longer notice losing that.
    assert!(rel < 1e-6, "scat μ rel error {rel:.3e} exceeds 1e-6");
}

/// Raw-scale robustness via the **Rust API** (no Python-layer standardization).
/// scat with identity link is location-scale equivariant, so scaling the
/// response by a large constant `S` must scale the fitted μ by exactly `S`.
///
/// Before scat's standardization was relocated into the fit core, a raw-scale
/// response made the inner solve's `X'WX + λS` ill-conditioned — `W = ½·Dμμ ~
/// 1/σ² ~ 1/S²` shrinks `X'WX` far below `λS`, so the Cholesky degenerates and
/// the outer Newton stalls. This test pins that the Rust API now fits large-`y`
/// scat correctly (the conditioning fix that previously only the Python wrapper
/// applied). The unit-scale companion is `scat_unweighted_n300_k10_cr` above.
#[test]
fn scat_raw_scale_via_rust_api() {
    let fx = load_fixture("1d_scat_unweighted_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let k = fx.inputs.k[0];

    // Scale the response up by 1e5 — comparable to a prices-style response
    // where var(y) ~ 1e10 and the raw-scale conditioning bug bites.
    const S: f64 = 1.0e5;
    let y_scaled = Array1::from_vec(fx.inputs.y_train.iter().map(|v| v * S).collect());
    let y_var = {
        let mean = y_scaled.iter().sum::<f64>() / (n as f64);
        y_scaled.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64)
    };
    // σ² seeded in *raw* (scaled) units, mirroring the unit-scale test's
    // var(y)/10 rough first-cut — the fit core standardizes internally.
    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y_scaled.view(),
        None,
        k,
    )
    .expect("gamrs::fit (raw-scale scat) should not fail");
    assert!(fit.converged, "raw-scale scat outer did not converge");

    let pred = fit.predict(x.view()).expect("predict should not fail");
    // Equivariance: μ̂(S·y) = S·μ̂(y), so pred/S must match the unit-scale
    // mgcv reference within the same bound as the unit-scale fit.
    let pred_unscaled: Vec<f64> = pred.iter().map(|p| p / S).collect();
    let rel = max_rel_err(&pred_unscaled, &fx.mgcv_output.predictions_train);
    println!(
        "[scat raw-scale S={S:.0e}] max_rel(pred/S vs mgcv) = {rel:.3e}; \
         σ̂² = {:.4e}; iters = {}",
        fit.scale, fit.n_iters,
    );
    // Bar 1e-6: observed 3.610e-8, against the unit-scale fit's 1.132e-7 —
    // same order, which is the point of the test (equivariance, so the two
    // should track). Deliberately the SAME bar as the unit-scale assertion so
    // that a conditioning loss shows up as this arm crossing a bar the other
    // one clears, rather than both being graded on their own curve.
    assert!(
        rel < 1e-6,
        "raw-scale scat μ rel error {rel:.3e} exceeds 1e-6 (conditioning?)"
    );
}

/// Regression for the v0.11 `scat` IRLS-fallback bug. The bug inverted the robustness: in the
/// presence of high-side outliers gamrs `scat` pulled the fitted mean
/// *above* the Gaussian/OLS fit, when scat must down-weight large-residual
/// points and pull the fit *below* it (toward the bulk). This is a
/// synthetic, self-contained restatement of the real-data Phoenix fixture
/// that triggered it — no proprietary data — so the directional contract
/// is locked in CI.
///
/// Construction: a gentle clean trend with a dense cluster of large
/// **positive** outliers. Gaussian chases them up; scat down-weights them.
/// The directional assertions (`scat < gaussian`, scat closer to the clean
/// trend) would both have FAILED under the pre-fix wrong-sign behaviour.
#[test]
fn scat_downweights_high_outliers_synthetic() {
    let n = 240usize;
    let x_vec: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    // Clean signal: gentle slope; mean over [0,1) ≈ 10.25.
    let signal = |xi: f64| 10.0 + 0.5 * xi;
    // Deterministic, ~zero-mean bounded "noise" (no RNG dependency).
    let noise = |i: usize| 0.05 * ((i as f64) * 0.7).sin();
    let mut y_vec: Vec<f64> = (0..n).map(|i| signal(x_vec[i]) + noise(i)).collect();
    // Inject large positive outliers on ~12.5% of rows, spread across x.
    for i in (0..n).step_by(8) {
        y_vec[i] += 8.0;
    }

    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(y_vec);
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let y_var = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n as f64);
    let k = 10usize;

    let g = gamrs::fit(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("gaussian fit should not fail");
    let s = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("scat fit should not fail");

    let mu_g = g.predict(x.view()).expect("gaussian predict");
    let mu_s = s.predict(x.view()).expect("scat predict");
    assert!(
        mu_s.iter().all(|v| v.is_finite()),
        "scat predictions must be finite"
    );

    let mean_g = mu_g.iter().sum::<f64>() / (n as f64);
    let mean_s = mu_s.iter().sum::<f64>() / (n as f64);
    let clean_level = 10.25; // mean of `signal` over x ∈ [0,1)
    println!(
        "[scat outlier-direction] gaussian_mean={mean_g:.3} scat_mean={mean_s:.3} \
         clean≈{clean_level:.3}; ν̂ via σ̂²={:.3e}; gap={:.3}",
        s.scale,
        mean_g - mean_s,
    );

    // (1) Correct sign: scat sits BELOW gaussian under positive outliers.
    //     Pre-fix this was inverted (scat ABOVE gaussian) — the bug.
    assert!(
        mean_s < mean_g,
        "scat mean {mean_s:.3} must be below gaussian {mean_g:.3} \
         under high-side outliers (robustness sign regression)"
    );
    // (2) Non-trivial gap: require a clear separation, not just noise.
    assert!(
        mean_g - mean_s > 0.2,
        "scat should down-weight outliers by a clear margin (gap {:.3} ≤ 0.2)",
        mean_g - mean_s
    );
    // (3) scat lands closer to the clean trend than the outlier-chasing
    //     Gaussian fit.
    assert!(
        (mean_s - clean_level).abs() < (mean_g - clean_level).abs(),
        "scat ({mean_s:.3}) should be closer to the clean level {clean_level:.3} \
         than gaussian ({mean_g:.3})"
    );
}
