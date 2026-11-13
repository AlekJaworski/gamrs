//! Per-phase profiling for 10-D Gaussian fit. Times each ScoreDerivatives
//! `value_grad_hess` call's three sub-phases (inner fit, value-grad,
//! hess_analytic) by reaching into `EnvelopeScore` directly.

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2};
use serde_json::Value;

use gamrs::design::{Additive, DesignStrategy, TermSpec};
use gamrs::score::GaussianClosedFormScore;
use gamrs::traits::{InnerSolver, ScoreDerivatives};

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

    let prep = Additive { terms }.prepare(x.view()).unwrap();
    let p = prep.x_design.ncols();
    println!("[profile] n={n} d={d} p={p}");

    let score = GaussianClosedFormScore::new(
        prep.x_design.clone(),
        y.clone(),
        prep.s_list.clone(),
        None,
        prep.rank_s_list.clone(),
        prep.mp,
        prep.log_pseudo_det_s_list.clone(),
    );

    // Probe theta values likely to be hit by the outer Newton: start at
    // zero (ρ_j = 0 → λ_j = 1) then march toward an over-smoothed regime.
    let mut probe_thetas: Vec<Array1<f64>> = vec![Array1::zeros(d)];
    for shift in [1.0_f64, 3.0, 5.0, 8.0] {
        let mut t = Array1::from_elem(d, shift);
        t[0] += 0.3;
        probe_thetas.push(t);
    }
    // Warm up.
    for theta in &probe_thetas {
        let _ = score.value_grad_hess(theta);
    }

    let mut sum_inner_us = 0.0_f64;
    let mut sum_vg_us = 0.0_f64;
    let mut sum_hess_us = 0.0_f64;
    let mut sum_total_us = 0.0_f64;
    let mut sum_inv_us = 0.0_f64;

    let runs = 30;
    for _ in 0..runs {
        for theta in &probe_thetas {
            let t_inner = Instant::now();
            let inner = score.inner.fit(theta).unwrap();
            let inner_us = t_inner.elapsed().as_secs_f64() * 1e6;

            let t_inv = Instant::now();
            let _ainv = inner.a_inv();
            let inv_us = t_inv.elapsed().as_secs_f64() * 1e6;

            let t_vg = Instant::now();
            let (_v, _g) = score.value_and_grad(theta).unwrap();
            let vg_us = t_vg.elapsed().as_secs_f64() * 1e6;

            let t_total = Instant::now();
            let (_, _, _h) = score.value_grad_hess(theta).unwrap();
            let total_us = t_total.elapsed().as_secs_f64() * 1e6;
            let hess_us = total_us - vg_us;

            sum_inner_us += inner_us;
            sum_inv_us += inv_us;
            sum_vg_us += vg_us;
            sum_hess_us += hess_us.max(0.0);
            sum_total_us += total_us;
            let _ = inner;
        }
    }
    let nprobe = (runs * probe_thetas.len()) as f64;
    println!(
        "[profile] per call avg (us): inner={:.1}  a_inv_alone={:.1}  value_and_grad={:.1}  hess(diff)={:.1}  total={:.1}",
        sum_inner_us / nprobe,
        sum_inv_us / nprobe,
        sum_vg_us / nprobe,
        sum_hess_us / nprobe,
        sum_total_us / nprobe,
    );
}
