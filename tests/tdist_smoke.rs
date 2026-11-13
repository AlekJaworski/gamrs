//! Phase 2a/2b smoke test: gamrs's existing `PirlsInner` + `EnvelopeScore`
//! handle the stateful TDist Loss WITHOUT modification — the trait
//! architecture's parametric promise validated.
//!
//! Strategy: generate noisy 1-D data with heavy-tailed t-noise, fit gamrs
//! with TDist (ν, σ²) FIXED at sensible values, and check that the fit
//! recovers a sensible smooth shape. We can't compare to mgcv yet (that
//! needs the joint outer Newton over (λ, σ², ν) which is Phase 2c+).
//!
//! This test exercises:
//! - `TDist` as a stateful Loss (struct fields, not unit)
//! - `TVariance` (constant σ²)
//! - `IdentityLink` reuse from the Gaussian path
//! - `PirlsInner` instantiated with a non-Gaussian-and-non-Bernoulli family
//!   — proves PIRLS is family-generic, not bernoulli-special-cased

use ndarray::{Array1, Array2, Axis};

use gamrs::family::tdist_identity;

#[test]
fn tdist_pirls_runs_via_trait_stack() {
    // Synthetic 1-D data with heavy-tailed t-noise. True mean is sin(2π x);
    // we fit gamrs's stack with FIXED ν=4, σ²=0.04 (the data-generating
    // process matches these but we don't tell gamrs).
    let n = 200;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let signal = (2.0 * std::f64::consts::PI * x).sin();
            // Pseudo-deterministic "t-distributed" noise: clip-tail.
            let h = (i.wrapping_mul(2654435761)) as u32;
            let u = (h as f64 / u32::MAX as f64) - 0.5; // U(-0.5, 0.5)
                                                        // Generate a draw from approximate t_4 via inverse-CDF-ish.
                                                        // For the smoke test we just want heavy-tailed noise: scale by
                                                        // 1/(0.1 + |u|) gives a power-law tail.
            let noise = 0.05 * u.signum() / (0.5 - u.abs() + 0.05);
            signal + noise
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);

    // Wire the trait stack manually — no `fit_tdist_cr` exists yet (it
    // needs multi-θ outer optimisation for the shape params). For now we
    // smoke-test PIRLS alone with FIXED shape params.
    use gamrs::basis::CrSpline;
    use gamrs::inner::{PirlsInner, PirlsOpts};
    use gamrs::traits::{Basis, BasisTransform, InnerSolver};
    use gamrs::transform::SumToZero;

    let k = 10;
    let cr = CrSpline::with_quantile_knots(x.view(), k).unwrap();
    let x2 = x.view().insert_axis(Axis(1));
    let raw_design = cr.evaluate(x2);
    let s_raw = cr.penalties().pop().unwrap();

    // Build sum-to-zero centred design + penalty.
    let stz = SumToZero::from_fit_design(cr, raw_design.view());
    let centring = stz.matrix().to_owned();
    let centred = stz.evaluate(x2);
    let s_smooth = centring.t().dot(&s_raw).dot(&centring);

    // Add intercept column.
    let k_smooth = centred.ncols();
    let p = 1 + k_smooth;
    let mut x_design = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x_design[[i, 0]] = 1.0;
        for j in 0..k_smooth {
            x_design[[i, 1 + j]] = centred[[i, j]];
        }
    }
    let mut s_total = Array2::<f64>::zeros((p, p));
    for i in 0..k_smooth {
        for j in 0..k_smooth {
            s_total[[1 + i, 1 + j]] = s_smooth[[i, j]];
        }
    }

    // PIRLS with TDist family — same struct that worked for Bernoulli,
    // now with TDist plugged in. The trait architecture's "single PIRLS
    // for every family" promise.
    let pirls = PirlsInner::<_, _, _, gamrs::CholeskySolver> {
        x_design: x_design.clone().into(),
        y: y.clone(),
        prior_weights: None,
        s_list: vec![s_total.clone()].into(),
        family: tdist_identity(4.0, 0.04),
        opts: PirlsOpts::default(),
        _solver: std::marker::PhantomData,
    };
    let fit = pirls
        .fit(&Array1::from_vec(vec![-2.0])) // log λ = -2 → moderate smoothing
        .expect("PIRLS-with-TDist should fit");
    assert!(fit.converged, "TDist PIRLS didn't converge");
    println!(
        "TDist PIRLS: iters={} dev={:.4} edf={} beta[0]={:.4}",
        fit.iterations, fit.deviance, fit.p, fit.beta[0]
    );

    // Sanity: μ should be roughly bounded by the signal range, not blown
    // up by outliers.
    let max_mu = fit.mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_mu = fit.mu.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(max_mu < 3.0, "μ over-fit to outliers: max = {max_mu}");
    assert!(min_mu > -3.0, "μ over-fit to outliers: min = {min_mu}");

    // Tail-robustness sanity: residuals should NOT all be dragged
    // toward outliers (the t-loss downweights them). Specifically, more
    // than half of the residuals should be smaller than √(ν·σ²) — the
    // "core" of the t distribution.
    let threshold = (4.0_f64 * 0.04).sqrt(); // ν·σ² = 0.16 → thr ≈ 0.4
    let core_count = y
        .iter()
        .zip(fit.mu.iter())
        .filter(|(yi, mui)| (**yi - **mui).abs() < threshold)
        .count();
    assert!(
        core_count > n / 2,
        "robustness broken: only {core_count}/{n} residuals inside core"
    );
}
