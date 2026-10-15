//! Extrapolation behaviour for multi-dimensional smooths — additive
//! (`s(x0) + s(x1)`) and tensor-product (`te(x0, x1)`) CR fits.
//!
//! Extends the 1-D coverage in `extrapolation_behavior.rs`:
//!   - additive: η = intercept + s0(x0) + s1(x1); beyond the range in one axis
//!     (the other held in-range) each marginal continues linearly, so η is
//!     linear in the swept axis. Beyond range in both axes it stays finite.
//!   - tensor: the te basis is a product of the (linearly-extrapolated) margins
//!     — η is *bilinear* off the corner, not linear, so we assert the key
//!     robustness guarantee (finite, no NaN) rather than a shape.
//!   - log link: μ = exp(η) stays > 0 at moderate extrapolation, never
//!     NaN/negative far out.

use std::f64::consts::PI;

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, MarginKind, TermSpec};
use gamrs::family::{gaussian_identity, poisson_log};
use gamrs::fit_with_design;

fn grid2(pairs: &[(f64, f64)]) -> Array2<f64> {
    let n = pairs.len();
    let mut flat = Vec::with_capacity(n * 2);
    for &(a, b) in pairs {
        flat.push(a);
        flat.push(b);
    }
    Array2::from_shape_vec((n, 2), flat).unwrap()
}

/// Decorrelated 2-D training inputs on [0,1]²: x0 monotone, x1 golden-ratio
/// jitter (so the two additive terms are identifiable, not collinear).
fn train_2d(n: usize) -> (Array2<f64>, Vec<f64>, Vec<f64>) {
    let x0: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let x1: Vec<f64> = (0..n)
        .map(|i| ((i as f64) * 0.618_033_988_75).fract())
        .collect();
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        flat.push(x0[i]);
        flat.push(x1[i]);
    }
    (Array2::from_shape_vec((n, 2), flat).unwrap(), x0, x1)
}

fn assert_linear(seq: &[f64], ctx: &str) {
    let step = seq[1] - seq[0];
    let max2 = seq
        .windows(3)
        .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
        .fold(0.0_f64, f64::max);
    let rel = max2 / (step.abs() + 1e-12);
    assert!(
        rel < 1e-6,
        "{ctx}: not linear — max|2nd diff|={max2:.3e}, rel={rel:.3e}"
    );
}

fn assert_all_finite(seq: &[f64], ctx: &str) {
    assert!(
        seq.iter().all(|v| v.is_finite()),
        "{ctx}: non-finite: {seq:?}"
    );
}

#[test]
fn additive_identity_extrapolates_linearly_along_one_axis() {
    let n = 400;
    let (x, x0, x1) = train_2d(n);
    let ys: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * x0[i]).sin() + 0.5 * (2.0 * PI * x1[i]).cos())
        .collect();
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 8,
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 8,
            pc: None,
            by: None,
        },
    ];
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        Array1::from(ys).view(),
        None,
    )
    .expect("additive gaussian fit");

    // Sweep x0 beyond its range with x1 held mid-range: η = c + s0(x0) + s1(0.5)
    // is linear in x0 (s0 extrapolates linearly, s1(0.5) is constant).
    let probes: Vec<(f64, f64)> = [1.2, 1.4, 1.6, 1.8, 2.0]
        .iter()
        .map(|&a| (a, 0.5))
        .collect();
    let eta = fit.predict(grid2(&probes).view()).unwrap();
    assert_linear(eta.as_slice().unwrap(), "additive x0-extrap (x1 fixed)");
    assert_all_finite(eta.as_slice().unwrap(), "additive x0-extrap");
}

#[test]
fn additive_finite_both_axes_beyond_range() {
    let n = 400;
    let (x, x0, x1) = train_2d(n);
    let ys: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * x0[i]).sin() + 0.5 * (2.0 * PI * x1[i]).cos())
        .collect();
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 8,
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 8,
            pc: None,
            by: None,
        },
    ];
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        Array1::from(ys).view(),
        None,
    )
    .expect("additive gaussian fit");

    let probes = [
        (1.5, 1.5),
        (2.0, -1.0),
        (-0.5, 1.8),
        (-2.0, -2.0),
        (50.0, 50.0),
        (-50.0, -50.0),
    ];
    let eta = fit.predict(grid2(&probes).view()).unwrap();
    assert_all_finite(eta.as_slice().unwrap(), "additive both-axes-beyond");
}

#[test]
fn additive_log_link_positive_mu_beyond_range() {
    let n = 300;
    let (x, x0, x1) = train_2d(n);
    // Deterministic positive counts.
    let ys: Vec<f64> = (0..n)
        .map(|i| {
            (0.3 + 0.5 * (2.0 * PI * x0[i]).sin() + 0.3 * x1[i])
                .exp()
                .round()
        })
        .collect();
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 6,
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 6,
            pc: None,
            by: None,
        },
    ];
    let fit = fit_with_design(
        poisson_log(),
        Additive { terms },
        x.view(),
        Array1::from(ys).view(),
        None,
    )
    .expect("additive poisson fit");

    // Moderate extrapolation: μ = exp(η) finite and > 0.
    let moderate = [(1.2, 0.5), (1.4, 0.5), (-0.2, 0.5), (0.5, 1.2)];
    let eta = fit.predict(grid2(&moderate).view()).unwrap();
    let mu: Vec<f64> = eta.iter().map(|&e| fit.link_kind.inverse(e)).collect();
    assert_all_finite(&mu, "additive poisson μ moderate");
    assert!(
        mu.iter().all(|&m| m > 0.0),
        "additive poisson μ must be >0: {mu:?}"
    );

    // Far out: never NaN/negative (may overflow to +inf, gracefully).
    let far: Vec<f64> = fit
        .predict(grid2(&[(20.0, 20.0), (-20.0, -20.0)]).view())
        .unwrap()
        .iter()
        .map(|&e| fit.link_kind.inverse(e))
        .collect();
    assert!(
        far.iter().all(|&m| !m.is_nan() && m >= 0.0),
        "additive poisson μ far: {far:?}"
    );
}

#[test]
fn tensor_te_finite_beyond_2d_range() {
    let n = 300;
    let (x, x0, x1) = train_2d(n);
    let ys: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * x0[i]).sin() * (2.0 * PI * x1[i]).cos())
        .collect();
    let terms = vec![TermSpec::Tensor {
        col_a: 0,
        col_b: 1,
        k_a: 5,
        k_b: 5,
        bs_a: MarginKind::Cr,
        bs_b: MarginKind::Cr,
    }];
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        Array1::from(ys).view(),
        None,
    )
    .expect("tensor te fit");

    // te extrapolation is bilinear off the data corner — assert it stays finite
    // (no NaN/inf) at the out-of-range corners and arbitrarily far out.
    let probes = [
        (1.5, 1.5),
        (2.0, 0.5),
        (0.5, 2.0),
        (-0.5, -0.5),
        (1.5, -0.5),
        (10.0, 10.0),
        (-10.0, -10.0),
    ];
    let pred = fit.predict(grid2(&probes).view()).unwrap();
    assert_all_finite(pred.as_slice().unwrap(), "tensor te beyond 2-D range");
}
