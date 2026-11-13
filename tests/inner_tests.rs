//! Inner-solver unit tests — closed-form Gaussian and PIRLS Bernoulli
//! basic correctness. Lifted out of `src/inner.rs` to keep the source
//! module under the project's >700-LOC threshold
//! (architecture-assumptions.md §G).

use approx::assert_relative_eq;
use gamrs::family::bernoulli_logit;
use gamrs::inner::{gaussian_inner_solve, CholeskySolver, PirlsInner, PirlsOpts};
use gamrs::traits::InnerSolver;
use ndarray::{array, Array2};
use std::marker::PhantomData;

#[test]
fn unpenalized_recovers_ols() {
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let y = array![1.0, 2.5, 4.0, 5.5];
    let s = Array2::<f64>::zeros((2, 2));
    // λ = 0 → s_total = 0·S = 0.
    let s_total = Array2::<f64>::zeros((2, 2));
    let _ = s; // unused now (only the scaled s_total feeds the solver)
    let fit =
        gaussian_inner_solve::<CholeskySolver>(x.view(), y.view(), None, s_total.view()).unwrap();
    // OLS β = [1.0, 1.5].
    assert_relative_eq!(fit.beta[0], 1.0, max_relative = 1e-10);
    assert_relative_eq!(fit.beta[1], 1.5, max_relative = 1e-10);
}

#[test]
fn pirls_bernoulli_recovers_logistic_fit_on_noisy_data() {
    // Logistic regression on noisy binary data. We flipped y so x<0 → y=1
    // (with one mislabelled point each side to avoid perfect separation
    // which would push β to infinity and singularise X'WX).
    let x = array![
        [1.0, -2.0],
        [1.0, -1.5],
        [1.0, -1.0],
        [1.0, -0.5],
        [1.0, 0.5],
        [1.0, 1.0],
        [1.0, 1.5],
        [1.0, 2.0],
        [1.0, -1.2], // mislabelled
        [1.0, 1.2],  // mislabelled
    ];
    let y = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    // Tiny ridge on the slope to prevent perfect separation (defensive).
    let s = array![[0.0, 0.0], [0.0, 1.0]];
    let solver = PirlsInner::<_, _, _, CholeskySolver> {
        x_design: x.into(),
        y,
        prior_weights: None,
        s_list: vec![s].into(),
        family: bernoulli_logit(),
        opts: PirlsOpts::default(),
        _solver: PhantomData,
    };
    let fit = solver.fit(&array![-5.0]).unwrap(); // tiny but nonzero λ
    assert!(
        fit.beta[1] < 0.0,
        "slope should be negative; got {}",
        fit.beta[1]
    );
    assert!(fit.converged, "PIRLS did not converge");
    assert_eq!(fit.eta.len(), 10);
    assert_eq!(fit.mu.len(), 10);
    assert_eq!(fit.working_weights.len(), 10);
    assert!(fit.working_weights.iter().all(|&w| w > 0.0));
    for &m in fit.mu.iter() {
        assert!(m > 0.0 && m < 1.0, "μ = {m} out of bounds");
    }
}

#[test]
fn ridge_shrinks_toward_zero() {
    let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let y = array![1.0, 2.0, 3.0];
    let s = array![[0.0, 0.0], [0.0, 1.0]]; // penalize slope only
    let s_low = &s * 1e-6_f64;
    let s_high = &s * 1000.0_f64;
    let fit_low =
        gaussian_inner_solve::<CholeskySolver>(x.view(), y.view(), None, s_low.view()).unwrap();
    let fit_high =
        gaussian_inner_solve::<CholeskySolver>(x.view(), y.view(), None, s_high.view()).unwrap();
    assert!(fit_high.beta[1].abs() < fit_low.beta[1].abs());
}
