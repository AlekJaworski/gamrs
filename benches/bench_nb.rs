//! Wall-time benchmark: NegBin profile-θ fits (1-D + 2-D additive).
//! Used to track the outer-Newton iter-count and per-iter cost vs
//! mgcv_rust's NegBin path (`src/smooth.rs:3562-3637`).

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::negbin_log;
use gamrs::fit_with_design;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
}
#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
}

fn load(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn bench_1d() {
    let fx = load("1d_nb_log_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    // Warm up
    for _ in 0..3 {
        let _ = gamrs::fit(gamrs::family::negbin_log(5.0), x.view(), y.view(), None, k).unwrap();
    }
    let mut times: Vec<f64> = Vec::new();
    let mut iters_v = 0;
    for _ in 0..30 {
        let t = Instant::now();
        let fit = gamrs::fit(gamrs::family::negbin_log(5.0), x.view(), y.view(), None, k).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        iters_v = fit.n_iters;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    println!(
        "[1D NB n=300 k=10] median={:.2}ms min={:.2}ms iters={}",
        median, min, iters_v
    );
}

fn bench_2d() {
    let fx = load("2d_nb_log_n600_k8_cr");
    let n = fx.inputs.x_train.len();
    let d = fx.inputs.x_train[0].len();
    let x_flat: Vec<f64> = fx
        .inputs
        .x_train
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect();
    let x = Array2::from_shape_vec((n, d), x_flat).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let ks: Vec<usize> = fx.inputs.k.clone();

    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: ks[0],
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: ks[1],
            pc: None,
            by: None,
        },
    ];

    // Warm
    for _ in 0..3 {
        let _ = fit_with_design(
            negbin_log(5.0),
            Additive {
                terms: terms.clone(),
            },
            x.view(),
            y.view(),
            None,
        )
        .unwrap();
    }
    let mut times: Vec<f64> = Vec::new();
    let mut iters_v = 0;
    for _ in 0..20 {
        let t = Instant::now();
        let fit = fit_with_design(
            negbin_log(5.0),
            Additive {
                terms: terms.clone(),
            },
            x.view(),
            y.view(),
            None,
        )
        .unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        iters_v = fit.n_iters;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    println!(
        "[2D NB n=600 k=8x2] median={:.2}ms min={:.2}ms iters={}",
        median, min, iters_v
    );
}

/// Synthetic 2-D additive NegBin at a realistic size. The fixture-backed
/// `bench_2d` (n=600) is too small to expose the multi-smooth profile-θ
/// scaling gap — bench_matters.py reports the embarrassing 0.06× vs
/// mgcv_rust at n=2K, not at n=600. This generates deterministic count data
/// (no RNG, no fixture) so the wall time at n∈{2K,5K} is observable.
fn bench_2d_synthetic(n: usize, warm: usize, reps: usize) {
    use std::f64::consts::PI;
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * 2);
    let mut y_vec: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let x0 = i as f64 / n as f64;
        let x1 = (i as f64 * 0.618_033_988_75).fract(); // golden-ratio scatter
        x_flat.push(x0);
        x_flat.push(x1);
        let mu = (1.0 + 1.2 * (2.0 * PI * x0).sin().abs() + 0.8 * x1).exp();
        let u = (i as f64 * 12.9898).sin().abs(); // deterministic [0,1)
        y_vec.push((mu * (0.5 + u)).round().max(0.0));
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();
    let y = Array1::from_vec(y_vec);
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 10,
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 10,
            pc: None,
            by: None,
        },
    ];
    let run = || {
        fit_with_design(
            negbin_log(5.0),
            Additive {
                terms: terms.clone(),
            },
            x.view(),
            y.view(),
            None,
        )
        .unwrap()
    };
    for _ in 0..warm {
        let _ = run();
    }
    let mut times: Vec<f64> = Vec::new();
    let mut iters_v = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let fit = run();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        iters_v = fit.n_iters;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    println!(
        "[2D NB synthetic n={n} k=10x2] median={:.1}ms min={:.1}ms iters={}",
        median, times[0], iters_v
    );
}

fn main() {
    println!("[bench_nb] NegBin profile-θ benchmarks");
    bench_1d();
    // Per-phase breakdown of the 2D fit: build with `--features profile`
    // and the reset/dump below emit the phase table (rho_only_total /
    // fit_inner_pirls / frozen_beta_probe / no_refresh_probe / hess_ift_rho).
    // No-ops without the feature.
    gamrs::profile::reset();
    bench_2d();
    gamrs::profile::dump(&mut std::io::stderr()).unwrap();
    // Realistic-size scaling check (release build recommended).
    bench_2d_synthetic(2000, 1, 3);
    bench_2d_synthetic(5000, 1, 3);
}
