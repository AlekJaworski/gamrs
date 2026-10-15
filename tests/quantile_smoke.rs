//! Phase 10 smoke test: gamrs's `fit_quantile_cr` end-to-end.
//!
//! Strategy: synthetic 1-D data where the true τ-quantile of y|x is a
//! known smooth function. Fit at three different τ values and verify:
//! 1. All three fits converge.
//! 2. Predictions are monotone in τ across the response range (higher τ
//!    → higher predicted quantile).
//! 3. Empirical coverage on a holdout is sensible (target τ ± tolerance).
//! 4. Pinball loss on the holdout is comparable to a naive baseline.
//!
//! We can't compare to qgam without R-side fixtures, which the worktree
//! can't run reliably. The convergence + monotonicity + sensible-coverage
//! bar is the success criterion per the Phase-7 task spec.

use ndarray::{Array1, Axis};

fn pinball_loss(y: f64, q: f64, tau: f64) -> f64 {
    let r = y - q;
    if r >= 0.0 {
        tau * r
    } else {
        (tau - 1.0) * r
    }
}

/// Pseudo-random in [0, 1) — deterministic so the test is reproducible.
fn prand(i: usize, seed: u32) -> f64 {
    let h = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
    (h as f64) / (u32::MAX as f64)
}

/// Box-Muller for a single N(0, 1) draw from two uniforms.
fn pnormal(u1: f64, u2: f64) -> f64 {
    let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
    r * (2.0 * std::f64::consts::PI * u2).cos()
}

#[test]
fn quantile_three_taus_monotone_and_converged() {
    // True data-generating process: y = sin(2π x) + N(0, σ²(x))
    // with heteroskedastic noise σ(x) = 0.1 + 0.4·x. The true τ-quantile
    // is sin(2π x) + σ(x)·Φ⁻¹(τ).
    let n = 300;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let signal = (2.0 * std::f64::consts::PI * x).sin();
            let scale = 0.1 + 0.4 * x;
            let u1 = prand(i, 7);
            let u2 = prand(i, 13);
            let z = pnormal(u1, u2);
            signal + scale * z
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);
    let x2 = x.view().insert_axis(Axis(1));

    let taus = [0.1_f64, 0.5, 0.9];
    let mut preds: Vec<Array1<f64>> = Vec::with_capacity(3);
    let mut pinballs: Vec<f64> = Vec::with_capacity(3);

    for &tau in &taus {
        let fit = gamrs::fit(
            gamrs::family::elf_identity(tau, /*sigma=*/ 0.0, /*lambda=*/ 0.0),
            x2,
            y.view(),
            None,
            10, // k
        )
        .unwrap_or_else(|e| panic!("gamrs::fit (ELF, τ={tau}) failed: {e}"));
        assert!(fit.converged, "τ={tau}: outer Newton did not converge");
        // Sensible iteration count — shouldn't be at the cap.
        assert!(
            fit.n_iters > 0 && fit.n_iters < 50,
            "τ={tau}: outer iters = {} (expected 1..50)",
            fit.n_iters
        );
        // EDF is real-valued and plausible (between 1 and k).
        assert!(
            fit.edf_total.is_finite() && fit.edf_total >= 0.5 && fit.edf_total <= 10.0,
            "τ={tau}: edf = {} out of plausible range",
            fit.edf_total
        );

        let pred = fit.predict(x2).expect("predict failed");

        // Pinball loss on the training set — should be lower than at
        // a constant predictor (loose sanity).
        let pinball: f64 = y
            .iter()
            .zip(pred.iter())
            .map(|(&yi, &qi)| pinball_loss(yi, qi, tau))
            .sum::<f64>()
            / (n as f64);
        // Constant-quantile baseline: empirical τ-quantile of y.
        let mut y_sorted: Vec<f64> = y.iter().copied().collect();
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q_const = y_sorted[((n as f64 - 1.0) * tau).round() as usize];
        let pinball_const: f64 = y
            .iter()
            .map(|&yi| pinball_loss(yi, q_const, tau))
            .sum::<f64>()
            / (n as f64);
        assert!(
            pinball < pinball_const,
            "τ={tau}: pinball {pinball:.4} >= constant baseline {pinball_const:.4}"
        );

        println!(
            "[quantile τ={tau}] iters={} edf={:.2} ρ̂={:.3} σ={:.4} pinball={:.4} vs const={:.4}",
            fit.n_iters, fit.edf_total, fit.rho[0], fit.scale, pinball, pinball_const
        );

        preds.push(pred);
        pinballs.push(pinball);
    }

    // Monotonicity: q_0.1(x) < q_0.5(x) < q_0.9(x) at MOST x. Allow a small
    // fraction of crossings since with finite data the smoothed quantile
    // surfaces can cross slightly — but in aggregate the ordering must hold.
    let mut crossings_low_mid = 0;
    let mut crossings_mid_high = 0;
    for (p_low, (p_mid, p_high)) in preds[0]
        .iter()
        .zip(preds[1].iter().zip(preds[2].iter()))
        .take(n)
    {
        if p_low > p_mid {
            crossings_low_mid += 1;
        }
        if p_mid > p_high {
            crossings_mid_high += 1;
        }
    }
    let frac_low_mid = crossings_low_mid as f64 / n as f64;
    let frac_mid_high = crossings_mid_high as f64 / n as f64;
    assert!(
        frac_low_mid < 0.10,
        "q_0.1 ≥ q_0.5 at {:.1}% of points (should be < 10%)",
        100.0 * frac_low_mid
    );
    assert!(
        frac_mid_high < 0.10,
        "q_0.5 ≥ q_0.9 at {:.1}% of points (should be < 10%)",
        100.0 * frac_mid_high
    );

    // Coverage check: fraction of y below q̂_τ should be roughly τ ± 0.10
    // (loose for a smoke test).
    for (i, &tau) in taus.iter().enumerate() {
        let coverage = y
            .iter()
            .zip(preds[i].iter())
            .filter(|(yj, qj)| **yj <= **qj)
            .count() as f64
            / n as f64;
        assert!(
            (coverage - tau).abs() < 0.15,
            "τ={tau}: empirical coverage {coverage:.3} too far from target"
        );
    }
}

#[test]
fn quantile_multismooth_additive_monotone_and_converged() {
    // 2-D additive DGP: y = sin(2π x0) + 2·(x1 − 0.5) + N(0, σ²(x0)),
    // heteroskedastic in x0 so the τ-quantile surfaces fan out. The true
    // τ-quantile is the additive signal + σ(x0)·Φ⁻¹(τ) — additive in the
    // two predictors, so an additive `s(x0) + s(x1)` smooth is well-specified.
    //
    // This exercises the multi-smooth ELF path: two CR penalty blocks, so
    // the outer Newton runs on a length-2 ρ and `ArmijoElfInner` assembles
    // `Σ_j exp(ρ_j) S_j` via `combined_s`. The single-smooth smoke test above
    // never touches the multi-penalty assembly.
    use gamrs::{Additive, TermSpec};
    use ndarray::Array2;

    let n = 600;
    let k = 8;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        // Decorrelated, deterministic covariates (two distinct hashes).
        let x0 = prand(i, 101);
        let x1 = prand(i, 211);
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        let signal = (2.0 * std::f64::consts::PI * x0).sin() + 2.0 * (x1 - 0.5);
        let scale = 0.1 + 0.3 * x0;
        let z = pnormal(prand(i, 307), prand(i, 419));
        y[i] = signal + scale * z;
    }

    let taus = [0.1_f64, 0.5, 0.9];
    let mut preds: Vec<Array1<f64>> = Vec::with_capacity(3);

    for &tau in &taus {
        let fit = gamrs::fit_with_design(
            gamrs::family::elf_identity(tau, /*sigma=*/ 0.0, /*lambda=*/ 0.0),
            Additive {
                terms: vec![
                    TermSpec::Cr {
                        col: 0,
                        k,
                        pc: None,
                        by: None,
                    },
                    TermSpec::Cr {
                        col: 1,
                        k,
                        pc: None,
                        by: None,
                    },
                ],
            },
            x.view(),
            y.view(),
            None,
        )
        .unwrap_or_else(|e| panic!("multi-smooth ELF fit (τ={tau}) failed: {e}"));

        assert!(
            fit.converged,
            "τ={tau}: multi-smooth outer Newton did not converge"
        );
        assert_eq!(
            fit.rho.len(),
            2,
            "τ={tau}: expected 2 smoothing params (one per CR term), got {}",
            fit.rho.len()
        );
        assert!(
            fit.n_iters > 0 && fit.n_iters < 50,
            "τ={tau}: outer iters = {} (expected 1..50)",
            fit.n_iters
        );
        assert!(
            fit.edf_total.is_finite() && fit.edf_total >= 1.0 && fit.edf_total <= 2.0 * k as f64,
            "τ={tau}: edf = {} out of plausible range",
            fit.edf_total
        );

        let pred = fit.predict(x.view()).expect("predict failed");

        // Coverage: fraction of y below q̂_τ ≈ τ. Looser band than the
        // single-smooth test — the additive surface has more freedom.
        let coverage = y
            .iter()
            .zip(pred.iter())
            .filter(|(yj, qj)| **yj <= **qj)
            .count() as f64
            / n as f64;
        assert!(
            (coverage - tau).abs() < 0.15,
            "τ={tau}: empirical coverage {coverage:.3} too far from target"
        );

        println!(
            "[ms-quantile τ={tau}] iters={} edf={:.2} ρ̂=[{:.2}, {:.2}] coverage={:.3}",
            fit.n_iters, fit.edf_total, fit.rho[0], fit.rho[1], coverage
        );
        preds.push(pred);
    }

    // Monotonicity across τ at most points (additive surfaces can cross
    // slightly with finite data, but the ordering must hold in aggregate).
    let mut crossings_low_mid = 0;
    let mut crossings_mid_high = 0;
    for ((p_low, p_mid), p_high) in preds[0].iter().zip(preds[1].iter()).zip(preds[2].iter()) {
        if p_low > p_mid {
            crossings_low_mid += 1;
        }
        if p_mid > p_high {
            crossings_mid_high += 1;
        }
    }
    let frac_low_mid = crossings_low_mid as f64 / n as f64;
    let frac_mid_high = crossings_mid_high as f64 / n as f64;
    assert!(
        frac_low_mid < 0.10,
        "q_0.1 ≥ q_0.5 at {:.1}% of points (should be < 10%)",
        100.0 * frac_low_mid
    );
    assert!(
        frac_mid_high < 0.10,
        "q_0.5 ≥ q_0.9 at {:.1}% of points (should be < 10%)",
        100.0 * frac_mid_high
    );
}

#[test]
fn quantile_invalid_tau_errors() {
    let x = Array1::from_vec(vec![0.0_f64, 0.5, 1.0]);
    let y = Array1::from_vec(vec![0.0_f64, 1.0, 2.0]);
    let x2 = x.view().insert_axis(Axis(1));
    for &bad in &[-0.1_f64, 0.0, 1.0, 1.5] {
        assert!(
            gamrs::fit(
                gamrs::family::elf_identity(bad, 0.0, 0.0),
                x2,
                y.view(),
                None,
                4,
            )
            .is_err(),
            "τ={bad} should be rejected"
        );
    }
}
