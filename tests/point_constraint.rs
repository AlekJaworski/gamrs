//! mgcv's point constraint `s(x, pc = v)` — the smooth passes through zero
//! at `v` and the intercept takes up the difference.
//!
//! What makes this feature testable is what it does NOT change. `pc` swaps
//! one identifiability constraint row for another (`mgcv:::smooth.construct3`
//! sets `object$C <- Predict.matrix(object, pc)$X`), and the CR penalty
//! annihilates constants, so the model space, λ̂, the fitted values and the
//! edf are all invariant — only the split between the intercept and the
//! smooth moves. A caller reading a *partial* curve sees that move; a caller
//! reading predictions cannot. Both halves are asserted here, because
//! "predictions unchanged" is the claim most easily mistaken for "the
//! constraint was ignored" (which is what gamrs did until this test existed).

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, Predictor, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::{fit_with_design, FittedGam};

/// Two decorrelated columns on [0, 1]: x0 monotone, x1 golden-ratio jitter.
fn train(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut flat = Vec::with_capacity(n * 2);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = ((i as f64) * 0.618_033_988_75).fract();
        flat.push(x0);
        flat.push(x1);
        // Deterministic wiggle instead of noise — the constraint claim is
        // about the parameterisation, not about signal recovery.
        y[i] = 3.0 * (6.0 * x0).sin() + 2.0 * x1 * x1 + 0.05 * ((i % 7) as f64);
    }
    (Array2::from_shape_vec((n, 2), flat).unwrap(), y)
}

/// Contribution of term `j` (the smooth alone, no intercept) at `x_new`.
fn term_contribution(fit: &FittedGam, x_new: &Array2<f64>, j: usize) -> Array1<f64> {
    let Predictor::Additive(additive) = &fit.predictor else {
        panic!("term_contribution expects an additive predictor");
    };
    let ranges = &additive.term_col_ranges;
    let (start, end) = ranges[j];
    let design = fit.predictor.design(x_new.view()).expect("design");
    design
        .slice(ndarray::s![.., start..end])
        .dot(&fit.beta.slice(ndarray::s![start..end]))
}

fn fit_two_terms(x: &Array2<f64>, y: &Array1<f64>, pc: Option<f64>) -> FittedGam {
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 8,
            pc: None,
            by: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 6,
            pc,
            by: None,
        },
    ];
    fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("fit")
}

fn max_abs(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn pc_moves_the_intercept_and_leaves_the_fit_where_it_was() {
    let (x, y) = train(400);
    let v = 0.35_f64;
    let plain = fit_two_terms(&x, &y, None);
    let pinned = fit_two_terms(&x, &y, Some(v));

    // Same problem, so the same answer: fitted values, λ̂ and edf.
    let mu_plain = plain.predict(x.view()).unwrap();
    let mu_pinned = pinned.predict(x.view()).unwrap();
    let mu_scale = mu_plain.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mu_diff = max_abs(&mu_plain, &mu_pinned) / mu_scale;
    assert!(mu_diff < 1e-9, "fitted values moved: rel {mu_diff:.3e}");
    let edf_diff = (plain.edf_total - pinned.edf_total).abs() / plain.edf_total;
    assert!(edf_diff < 1e-8, "edf moved: rel {edf_diff:.3e}");
    for j in 0..plain.lambda.len() {
        let rel = (plain.lambda[j] - pinned.lambda[j]).abs() / plain.lambda[j];
        assert!(rel < 1e-6, "lambda[{j}] moved: rel {rel:.3e}");
    }

    // What did move: the constrained smooth is zero at v, and the intercept
    // absorbed exactly the level the plain fit's smooth carried there.
    let grid: Array2<f64> = {
        let pts: Vec<f64> = (0..21).map(|i| i as f64 / 20.0).collect();
        let mut flat = Vec::with_capacity(pts.len() * 2);
        for p in &pts {
            flat.push(0.5);
            flat.push(*p);
        }
        Array2::from_shape_vec((pts.len(), 2), flat).unwrap()
    };
    let at_v = Array2::from_shape_vec((1, 2), vec![0.5, v]).unwrap();
    let f_pinned_v = term_contribution(&pinned, &at_v, 1)[0];
    let f_plain_v = term_contribution(&plain, &at_v, 1)[0];
    let f_scale = term_contribution(&plain, &grid, 1)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        f_pinned_v.abs() < 1e-9 * f_scale,
        "smooth at pc is {f_pinned_v:.3e}, expected 0 (curve scale {f_scale:.3e})"
    );
    assert!(
        f_plain_v.abs() > 0.05 * f_scale,
        "test is vacuous: the plain smooth is already ~0 at v ({f_plain_v:.3e})"
    );
    let shift = pinned.beta[0] - plain.beta[0];
    assert!(
        (shift - f_plain_v).abs() < 1e-6 * f_scale,
        "intercept moved by {shift:.6} but the plain smooth at v is {f_plain_v:.6}"
    );

    // The whole curve is the plain curve minus that one constant.
    let curve_plain = term_contribution(&plain, &grid, 1);
    let curve_pinned = term_contribution(&pinned, &grid, 1);
    let expected = &curve_plain - f_plain_v;
    let curve_diff = max_abs(&curve_pinned, &expected) / f_scale;
    assert!(
        curve_diff < 1e-8,
        "constrained curve is not the plain curve shifted: rel {curve_diff:.3e}"
    );

    // The unconstrained neighbour term is untouched by its neighbour's pc.
    let other_diff = max_abs(
        &term_contribution(&plain, &grid, 0),
        &term_contribution(&pinned, &grid, 0),
    ) / f_scale;
    assert!(other_diff < 1e-8, "term 0 moved: rel {other_diff:.3e}");
}

#[test]
fn pc_holds_through_the_stable_reparam_rotation() {
    let (x, y) = train(300);
    let v = 0.6_f64;
    let terms = |pc| {
        vec![
            TermSpec::Cr {
                col: 0,
                k: 8,
                pc: None,
                by: None,
            },
            TermSpec::CrStable {
                col: 1,
                k: 6,
                pc,
                by: None,
            },
        ]
    };
    let plain = fit_with_design(
        gaussian_identity(),
        Additive { terms: terms(None) },
        x.view(),
        y.view(),
        None,
    )
    .expect("fit");
    let pinned = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: terms(Some(v)),
        },
        x.view(),
        y.view(),
        None,
    )
    .expect("fit");

    let mu_plain = plain.predict(x.view()).unwrap();
    let mu_pinned = pinned.predict(x.view()).unwrap();
    let mu_scale = mu_plain.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mu_diff = max_abs(&mu_plain, &mu_pinned) / mu_scale;
    assert!(mu_diff < 1e-9, "fitted values moved: rel {mu_diff:.3e}");

    let at_v = Array2::from_shape_vec((1, 2), vec![0.5, v]).unwrap();
    let f_pinned_v = term_contribution(&pinned, &at_v, 1)[0];
    let f_plain_v = term_contribution(&plain, &at_v, 1)[0];
    assert!(
        f_pinned_v.abs() < 1e-9 * f_plain_v.abs().max(1.0),
        "rotated smooth at pc is {f_pinned_v:.3e}, expected 0"
    );
}

/// Three columns: two for the smooths, plus a 0/1 indicator for a parametric term.
fn train_mixed(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut flat = Vec::with_capacity(n * 3);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = i as f64 / (n as f64 - 1.0);
        let x1 = ((i as f64) * 0.618_033_988_75).fract();
        let flag = if i % 3 == 0 { 1.0 } else { 0.0 };
        flat.push(x0);
        flat.push(x1);
        flat.push(flag);
        y[i] = 3.0 * (6.0 * x0).sin() + 2.0 * x1 * x1 + 5.0 * flag + 0.05 * ((i % 7) as f64);
    }
    (Array2::from_shape_vec((n, 3), flat).unwrap(), y)
}

#[test]
fn pc_on_a_smooth_leaves_a_parametric_term_alone() {
    // A parametric term has no centering constraint, so it has no null-space for the constraint
    // row to be absorbed into — which is exactly why it is worth asserting that the row does not
    // reach it. A pc that leaked into the parametric block would still zero the smooth at the
    // point and still leave fitted values invariant, so neither of those catches it; the
    // parametric coefficient has to be read directly.
    let (x, y) = train_mixed(400);
    let v = 0.35_f64;
    let terms = |pc| {
        vec![
            TermSpec::Cr {
                col: 0,
                k: 8,
                pc: None,
                by: None,
            },
            TermSpec::Cr {
                col: 1,
                k: 6,
                pc,
                by: None,
            },
            TermSpec::Parametric { col: 2 },
        ]
    };
    let fit_it = |pc| {
        fit_with_design(
            gaussian_identity(),
            Additive { terms: terms(pc) },
            x.view(),
            y.view(),
            None,
        )
        .expect("fit")
    };
    let plain = fit_it(None);
    let pinned = fit_it(Some(v));

    let Predictor::Additive(additive) = &plain.predictor else {
        panic!("expected an additive predictor");
    };
    let (p_start, p_end) = additive.term_col_ranges[2];
    assert_eq!(p_end - p_start, 1, "a parametric term is one column");

    // The claim. The indicator carries the whole +5.0 step, so a leak would be loud.
    let slope_plain = plain.beta[p_start];
    let slope_pinned = pinned.beta[p_start];
    assert!(
        slope_plain.abs() > 1.0,
        "test is vacuous: the parametric term carries no effect to preserve ({slope_plain:.3e})"
    );
    let slope_rel = (slope_pinned - slope_plain).abs() / slope_plain.abs();
    assert!(
        slope_rel < 1e-8,
        "the parametric coefficient moved with the pc: {slope_plain:.9} -> {slope_pinned:.9} (rel {slope_rel:.3e})"
    );

    // And the pc still does its job with a parametric term in the model.
    let at_v = Array2::from_shape_vec((1, 3), vec![0.5, v, 1.0]).unwrap();
    let f_plain_v = term_contribution(&plain, &at_v, 1)[0];
    let f_pinned_v = term_contribution(&pinned, &at_v, 1)[0];
    assert!(
        f_plain_v.abs() > 1e-3,
        "test is vacuous: the plain smooth is already ~0 at v ({f_plain_v:.3e})"
    );
    assert!(
        f_pinned_v.abs() < 1e-9 * f_plain_v.abs(),
        "smooth at pc is {f_pinned_v:.3e}, expected 0"
    );
    let shift = pinned.beta[0] - plain.beta[0];
    assert!(
        (shift - f_plain_v).abs() < 1e-6 * f_plain_v.abs(),
        "the intercept moved by {shift:.9}, but the plain smooth at v is {f_plain_v:.9} — with a \
         parametric term present the intercept must still absorb the whole difference"
    );

    // The fit itself is where it was.
    let mu_plain = plain.predict(x.view()).unwrap();
    let mu_pinned = pinned.predict(x.view()).unwrap();
    let mu_scale = mu_plain.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mu_diff = max_abs(&mu_plain, &mu_pinned) / mu_scale;
    assert!(mu_diff < 1e-9, "fitted values moved: rel {mu_diff:.3e}");
}
