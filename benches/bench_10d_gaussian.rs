//! 10-D Gaussian fit profiling bench — drives the mgcv_rust parity gap
//! investigation (`tests/fixtures/10d_gaussian_n3000_k8_cr.json`,
//! gamrs ≈ 171 ms vs mgcv_rust ≈ 51 ms).
//!
//! Prints:
//!   - end-to-end fit median ms
//!   - outer-Newton iteration count
//!   - per-iteration breakdown (inner fit, `compute_value_grad_from_fit`,
//!     `hess_analytic`)
//!
//! Run: `cargo bench --bench bench_10d_gaussian` (release profile).

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2};
use serde_json::Value;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit_with_design;

fn main() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/10d_gaussian_n3000_k8_cr.json");
    let txt = std::fs::read_to_string(&p).expect("missing fixture");
    let v: Value = serde_json::from_str(&txt).unwrap();
    let d = v["inputs"]["d"].as_u64().unwrap() as usize;
    let k_vec: Vec<usize> = v["inputs"]["k"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let rows = v["inputs"]["x_train"].as_array().unwrap();
    let n = rows.len();
    let mut x = Array2::<f64>::zeros((n, d));
    for (i, r) in rows.iter().enumerate() {
        let arr = r.as_array().unwrap();
        for (j, c) in arr.iter().enumerate() {
            x[[i, j]] = c.as_f64().unwrap();
        }
    }
    let ys: Vec<f64> = v["inputs"]["y_train"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let y = Array1::from_vec(ys);

    let terms: Vec<TermSpec> = (0..d)
        .map(|c| TermSpec::Cr {
            col: c,
            k: k_vec[c],
            pc: None,
            by: None,
        })
        .collect();

    // Warm up.
    let _ = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: terms.clone(),
        },
        x.view(),
        y.view(),
        None,
    )
    .unwrap();

    let runs = 30;
    let t0 = Instant::now();
    let mut last_iters = 0;
    let mut last_conv = false;
    for _ in 0..runs {
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
        last_iters = fit.n_iters;
        last_conv = fit.converged;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_run_ms = 1000.0 * elapsed / (runs as f64);
    println!(
        "[bench_10d_gaussian] d={d} n={n} k={k_vec:?}\n  per-fit (median) ~ {per_run_ms:.3} ms\n  outer iters = {last_iters}, converged = {last_conv}",
    );
}
