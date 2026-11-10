//! `scat` where a shape axis saturates — the regime behind the 0.14.x
//! non-convergence reports, end to end.
//!
//! WHAT THE REGIME IS. scat's outer Newton runs over `θ = [ρ, log σ², log(ν−3)]`
//! inside a box. Fit it to data that is NOT heavy-tailed and ν is unidentified:
//! its axis heads for the upper bound of 10 (ν ≈ 22029, i.e. Gaussian) and its
//! curvature goes NEGATIVE — the score has no interior optimum in that direction.
//! Both things then go wrong at once, and n = 66 with k = 12 makes the ρ ridge
//! flat enough to expose them:
//!
//!   1. the axis pins at the bound and its clamped-away step throttles the global
//!      per-axis cap shrink, so the axes that CAN move crawl at 1/20th speed;
//!   2. once concave, Newton's `-g/|H|` on it is ~0.02 per iteration against a
//!      distance of 2, so the loop marches ~100 iterations to a point that is
//!      known in closed form.
//!
//! Both ended in `NotConverged` at the 200-iteration cap, which used to DISCARD
//! the fit — `RuntimeError: solver did not converge after 200 iterations (last
//! grad norm = 5.646e-2)` from a single-smooth scat fit that 0.13.1 fit fine.
//!
//! WHERE THE MECHANISMS ARE PINNED. Precisely, at unit level, in
//! `src/outer/step.rs::tests` — `a_pinned_axis_does_not_throttle_the_live_ones`
//! reproduces the throttle arithmetic outright, and the `concave_bound_jump`
//! tests pin the jump rule. These tests are the end-to-end complement: they check
//! that a whole `scat` fit in this regime comes back usable, and they are the ones
//! that would catch a driver that stops calling those rules.
//!
//! Fixtures: generated here, deterministic (an LCG + Box-Muller), shaped after the
//! fit that surfaced this — n = 66, k = 12, a mild signal on a ~1e5 response
//! scale, light-tailed noise. No customer data.

use ndarray::{Array1, Array2};

/// n = 66 rows of `y = 550000 + 40000·x + quad·x² + 20000·z`, `x ~ U(-1, 1)`
/// sorted, `z` standard normal. `seed` selects the draw.
fn fixture(seed: u64, quad: f64) -> (Array2<f64>, Array1<f64>) {
    let n = 66usize;
    let mut state = seed;
    let mut lcg = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut xs: Vec<f64> = (0..n).map(|_| -1.0 + 2.0 * lcg()).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    let mut ys = Vec::with_capacity(n);
    for &x in &xs {
        let u1: f64 = lcg().max(1e-12);
        let u2: f64 = lcg();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        ys.push(550_000.0 + 40_000.0 * x + quad * x * x + 20_000.0 * z);
    }
    (
        Array2::from_shape_vec((n, 1), xs).unwrap(),
        Array1::from_vec(ys),
    )
}

fn fit(x: &Array2<f64>, y: &Array1<f64>) -> gamrs::FittedGam {
    let mean = y.iter().sum::<f64>() / (y.len() as f64);
    let y_var = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
    gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        12,
    )
    .expect("scat with a saturating shape axis must still return a fit")
}

/// The fitted curve has to track the generating signal, so that "converged" is a
/// claim about the answer and not just about the loop exiting.
fn max_rel_dev(x: &Array2<f64>, pred: &Array1<f64>, quad: f64) -> f64 {
    pred.iter()
        .zip(x.column(0).iter())
        .map(|(&p, &xi)| (p - (550_000.0 + 40_000.0 * xi + quad * xi * xi)).abs() / 550_000.0)
        .fold(0.0_f64, f64::max)
}

/// The three draws that used to raise `NotConverged` outright.
#[test]
fn saturating_shape_axis_fits_inside_the_budget() {
    for (seed, quad) in [(20u64, 0.0), (26, 15_000.0), (33, 15_000.0)] {
        let (x, y) = fixture(seed, quad);
        let f = fit(&x, &y);
        let pred = f.predict(x.view()).expect("predict");
        let dev = max_rel_dev(&x, &pred, quad);
        println!(
            "[seed {seed}/{quad}] iters={} conv={} edf={:.4} log(nu-3)={:.4} dev={dev:.3e}",
            f.n_iters, f.converged, f.edf_total, f.shape_params[1]
        );
        assert!(f.converged, "seed {seed}: must report convergence");
        // Measured 138 / 32 / 112. The bar has real headroom on purpose: these
        // draws are genuinely slow (a flat ρ ridge plus a saturating ν), and a bar
        // sitting just above the measurement turns every harmless change into a
        // false positive. Pre-fix all three hit the 200 cap and raised.
        assert!(
            f.n_iters < 180,
            "seed {seed}: took {} iterations — the throttle is back",
            f.n_iters
        );
        assert!(
            dev < 0.05,
            "seed {seed}: fitted curve is {dev:.3e} off the generating signal"
        );
    }
    // mgcv 1.9.4, `gam(y ~ s(x, bs="cr", k=12), family=scat(), method="REML")` on
    // the seed-26 draw: edf 3.048638. While ρ could take the concave bound jump it
    // was thrown to the upper face on this draw and came back a straight line, edf
    // 2.0000, on a signal with a 15,000 quadratic in it.
    let (x, y) = fixture(26, 15_000.0);
    let edf = fit(&x, &y).edf_total;
    assert!(
        (edf - 3.048638).abs() < 1e-3,
        "seed 26: edf {edf:.6}, mgcv 3.048638"
    );
}

/// The concave-crawl draw: 220 iterations before the jump rule, ~30 after. It is
/// the one that proved raising the budget is the wrong lever — those extra 130
/// iterations bought 1.3e-8 REML units and $19 on a $550k curve.
#[test]
fn indefinite_shape_axis_does_not_crawl_to_the_cap() {
    let (x, y) = fixture(21, 0.0);
    let f = fit(&x, &y);
    println!(
        "[indefinite] iters={} conv={} edf={:.4} nu={:.1}",
        f.n_iters,
        f.converged,
        f.edf_total,
        3.0 + f.shape_params[1].exp()
    );
    assert!(f.converged);
    assert!(
        f.n_iters < 100,
        "took {} iterations; the concave axis is crawling again",
        f.n_iters
    );
    let pred = f.predict(x.view()).expect("predict");
    assert!(max_rel_dev(&x, &pred, 0.0) < 0.05);
}

/// Aggregate lock over the whole regime: 39 draws x 3 signal shapes, every one a
/// saturating-ν scat fit. Catches a change that trades one seed's iterations for
/// another's, which a handful of named seeds cannot.
///
/// The iteration bar is set at the pre-fix total (4414), not just above the
/// measured 3866: the claim being locked is "no worse than before the fix", with
/// 14% headroom so an ordinary refactor is not a false positive.
#[test]
fn no_draw_in_the_regime_is_discarded() {
    let mut total = 0usize;
    let mut discarded = Vec::new();
    let mut non_converged = Vec::new();
    for seed in 1u64..40 {
        for &quad in &[0.0, 15_000.0, 40_000.0] {
            let (x, y) = fixture(seed, quad);
            let mean = y.iter().sum::<f64>() / (y.len() as f64);
            let y_var = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
            match gamrs::fit(
                gamrs::family::tdist_identity(5.0, y_var * 0.1),
                x.view(),
                y.view(),
                None,
                12,
            ) {
                Ok(f) => {
                    total += f.n_iters;
                    if !f.converged {
                        non_converged.push((seed, quad));
                    }
                }
                Err(_) => discarded.push((seed, quad)),
            }
        }
    }
    println!("[sweep] 117 fits, {total} outer iterations total (pre-fix: 4414)");
    assert!(
        discarded.is_empty(),
        "these draws were discarded rather than returned: {discarded:?}"
    );
    assert!(
        non_converged.is_empty(),
        "these draws came back non-converged: {non_converged:?}"
    );
    assert!(
        total < 4414,
        "117 fits took {total} outer iterations; pre-fix was 4414"
    );
}
