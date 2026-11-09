//! Smoke + equivalence tests for the n-margin tensor product `te(...)`
//! (`TermSpec::TeMulti`) and tensor interaction `ti(...)`
//! (`TermSpec::Ti`) paths.
//!
//! No mgcv n-margin te/ti fixture exists in the parity corpus (only a
//! 2-margin `2d_gaussian_te` fixture), so n-margin parity is covered by:
//!   1. a 3-margin Gaussian fit smoke test (converges, finite predictions,
//!      `rho.len() == 3`),
//!   2. a 2-margin equivalence check: `TeMulti` with 2 cols must produce
//!      the same fitted predictions (to ~1e-9) as the existing
//!      `TermSpec::Tensor` on identical data — proving the generalization
//!      reduces to the validated 2-margin path,
//!   3. a `ti` smoke test (converges, finite predictions).

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, MarginKind, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit_with_design;

/// Deterministic LCG so the test is reproducible without an RNG dep.
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

fn synth_3col(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
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
        // True surface with genuine 3-way structure + interactions.
        let f = (2.0 * std::f64::consts::PI * x0).sin()
            + (1.5 * x1 - 0.5).powi(2)
            + 0.8 * x0 * x2
            + 0.5 * x1 * x2;
        y[i] = f + noise;
    }
    (x, y)
}

fn synth_2col(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(0x0fee_1dad_dead_beef);
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = rng.next_f64();
        let x1 = rng.next_f64();
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        let noise = (rng.next_f64() - 0.5) * 0.15;
        let f = (2.0 * std::f64::consts::PI * x0).sin() + (1.5 * x1 - 0.5).powi(2) + 0.7 * x0 * x1;
        y[i] = f + noise;
    }
    (x, y)
}

#[test]
fn te_multi_3margin_gaussian_smoke() {
    let n = 400;
    let (x, y) = synth_3col(n);
    let terms = vec![TermSpec::TeMulti {
        cols: vec![0, 1, 2],
        k: vec![4, 4, 4],
        bs: vec![MarginKind::Cr, MarginKind::Cr, MarginKind::Cr],
    }];
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("3-margin TeMulti Gaussian fit failed");

    assert!(
        fit.converged,
        "outer Newton did not converge for 3-margin te"
    );
    assert_eq!(fit.rho.len(), 3, "one smoothing param per margin (D=3)");
    assert_eq!(fit.lambda.len(), 3);

    let pred = fit.predict(x.view()).expect("3-margin te predict failed");
    assert_eq!(pred.len(), n);
    for &p in pred.iter() {
        assert!(p.is_finite(), "non-finite prediction: {p}");
    }

    // Predict on a fresh grid to exercise the predictor rebuild.
    let mut x_new = Array2::<f64>::zeros((5, 3));
    for i in 0..5 {
        let t = 0.1 + 0.8 * (i as f64) / 4.0;
        x_new[[i, 0]] = t;
        x_new[[i, 1]] = 1.0 - t;
        x_new[[i, 2]] = 0.5;
    }
    let pred_new = fit
        .predict(x_new.view())
        .expect("3-margin te new-x predict failed");
    for &p in pred_new.iter() {
        assert!(p.is_finite(), "non-finite new-x prediction: {p}");
    }
}

#[test]
fn te_multi_2margin_matches_tensor() {
    let n = 350;
    let (x, y) = synth_2col(n);

    let fit_tensor = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: vec![TermSpec::Tensor {
                col_a: 0,
                col_b: 1,
                k_a: 5,
                k_b: 5,
                bs_a: MarginKind::Cr,
                bs_b: MarginKind::Cr,
                by: None,
            }],
        },
        x.view(),
        y.view(),
        None,
    )
    .expect("2-margin Tensor fit failed");

    let fit_multi = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: vec![TermSpec::TeMulti {
                cols: vec![0, 1],
                k: vec![5, 5],
                bs: vec![MarginKind::Cr, MarginKind::Cr],
            }],
        },
        x.view(),
        y.view(),
        None,
    )
    .expect("2-margin TeMulti fit failed");

    assert!(fit_tensor.converged && fit_multi.converged);
    assert_eq!(fit_multi.rho.len(), 2);

    let p_tensor = fit_tensor.predict(x.view()).expect("tensor predict");
    let p_multi = fit_multi.predict(x.view()).expect("multi predict");

    let max_diff = p_tensor
        .iter()
        .zip(p_multi.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("[te_multi_2margin_matches_tensor] max |TeMulti(2col) - Tensor| = {max_diff:.3e}");
    assert!(
        max_diff < 1e-9,
        "TeMulti with 2 cols must match Tensor to 1e-9, got {max_diff:.3e}"
    );
}

#[test]
fn ti_3margin_gaussian_smoke() {
    let n = 400;
    let (x, y) = synth_3col(n);
    let terms = vec![TermSpec::Ti {
        cols: vec![0, 1, 2],
        k: vec![4, 4, 4],
        bs: vec![MarginKind::Cr, MarginKind::Cr, MarginKind::Cr],
    }];
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("3-margin Ti Gaussian fit failed");

    assert!(fit.converged, "outer Newton did not converge for ti");
    assert_eq!(fit.rho.len(), 3, "one smoothing param per margin (D=3)");

    let pred = fit.predict(x.view()).expect("ti predict failed");
    assert_eq!(pred.len(), n);
    for &p in pred.iter() {
        assert!(p.is_finite(), "non-finite ti prediction: {p}");
    }
}
