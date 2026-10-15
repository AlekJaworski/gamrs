//! Wall-time benchmark: multi-smooth Gaussian fit through the analytic
//! outer-Newton Hessian path (landed in commit 49e9cc0, port of mgcv's
//! fastreml). Times end-to-end `fit_with_design(...)` on a synthetic
//! 3-smooth Cr-spline additive fixture.
//!
//! With the analytic Hessian, each outer iter costs ONE inner fit
//! (formula assembled from the converged inner state). The previous
//! central-FD path needed `1 + 2d` inner fits per outer iter — so for
//! d = 3 we expect ~7× fewer inner solves at the score level.
//!
//! Run: `cargo bench --bench bench_multi_smooth` (release profile).

use std::time::Instant;

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit_with_design;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn synth(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(seed);
    let mut x = Array2::<f64>::zeros((n, 3));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = rng.next_f64();
        let x1 = rng.next_f64();
        let x2 = rng.next_f64();
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        x[[i, 2]] = x2;
        let noise = (rng.next_f64() - 0.5) * 0.2;
        let f = (2.0 * std::f64::consts::PI * x0).sin()
            + (1.5 * x1 - 0.5).powi(2)
            + 0.5 * (3.0 * x2).cos();
        y[i] = f + noise;
    }
    (x, y)
}

fn run_one(n: usize) -> (f64, usize) {
    let (x, y) = synth(n, 0x0a11_ce00_1234_5678);
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
        TermSpec::Cr {
            col: 2,
            k: 10,
            pc: None,
            by: None,
        },
    ];

    // Warm-up.
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

    let runs = 100;
    let t0 = Instant::now();
    let mut last_iters = 0;
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
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_run_ms = 1000.0 * elapsed / (runs as f64);
    (per_run_ms, last_iters)
}

fn main() {
    println!("[bench_multi_smooth] 3-smooth Gaussian additive fit (Cr, k=10)");
    for &n in &[500usize, 2000, 5000] {
        let (ms, iters) = run_one(n);
        println!("  n={n:5}  {ms:8.3} ms/fit   outer_iters={iters}");
    }
}
