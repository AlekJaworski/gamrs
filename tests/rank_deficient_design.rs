//! A design whose coefficients the data and penalties cannot all identify
//! fits, with the aliased coefficient at exactly 0 — the fit is the one with
//! that column removed. Every case here used to fail with
//! "singular system: Cholesky failed".

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, TermSpec};
use gamrs::family::{bernoulli_logit, gaussian_identity};
use gamrs::fit_with_design;

const N: usize = 120;

/// Columns: 0 smooth covariate, 1 sparse tag, 2 a count in {1, 2}.
fn base() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..N).map(|i| ((i as f64) * 0.618_033_988_75).fract()).collect();
    let tag: Vec<f64> = (0..N).map(|i| if i % 17 == 3 { 1.0 } else { 0.0 }).collect();
    let count: Vec<f64> = (0..N).map(|i| if i % 9 == 0 { 1.0 } else { 2.0 }).collect();
    (x, tag, count)
}

fn columns(cols: &[&[f64]]) -> Array2<f64> {
    Array2::from_shape_fn((N, cols.len()), |(i, j)| cols[j][i])
}

fn response(x: &[f64], tag: &[f64], count: &[f64]) -> Array1<f64> {
    (0..N)
        .map(|i| (6.0 * x[i]).sin() + 0.5 * tag[i] + 0.3 * count[i] + 0.05 * ((i % 7) as f64))
        .collect()
}

fn terms(n_parametric: usize) -> Vec<TermSpec> {
    let mut t = vec![TermSpec::Cr { col: 0, k: 8, pc: None }];
    t.extend((1..=n_parametric).map(|col| TermSpec::Parametric { col }));
    t
}

#[test]
fn duplicate_tag_fits_as_if_the_copy_were_absent() {
    let (x, tag, count) = base();
    let y = response(&x, &tag, &count);
    let dup = fit_with_design(
        gaussian_identity(),
        Additive { terms: terms(3) },
        columns(&[&x, &tag, &count, &tag]).view(),
        y.view(),
        None,
    )
    .unwrap();
    let reduced = fit_with_design(
        gaussian_identity(),
        Additive { terms: terms(2) },
        columns(&[&x, &tag, &count]).view(),
        y.view(),
        None,
    )
    .unwrap();

    let p = dup.beta.len();
    assert_eq!(dup.aliased_coefficients(), vec![p - 1], "the later copy is the one zeroed");
    assert_eq!(dup.beta[p - 1], 0.0);
    assert!(dup.vcov.row(p - 1).iter().all(|&v| v == 0.0));
    assert!(reduced.aliased_coefficients().is_empty());
    for i in 0..p - 1 {
        assert!((dup.beta[i] - reduced.beta[i]).abs() < 1e-8, "coef {i}");
    }
    assert!((dup.reml_value - reduced.reml_value).abs() < 1e-8);
    assert!((dup.edf_total - reduced.edf_total).abs() < 1e-8);
}

#[test]
fn counts_summing_to_a_constant_keep_the_intercept() {
    // full + half == 2 on every row, so together they reproduce the intercept.
    let (x, tag, full) = base();
    let half: Vec<f64> = full.iter().map(|f| 2.0 - f).collect();
    let y = response(&x, &tag, &full);
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms: terms(3) },
        columns(&[&x, &tag, &full, &half]).view(),
        y.view(),
        None,
    )
    .unwrap();
    let p = fit.beta.len();
    assert_eq!(fit.aliased_coefficients(), vec![p - 1]);
    assert!(fit.beta[0] != 0.0);
}

#[test]
fn binomial_with_a_duplicate_tag_fits() {
    let (x, tag, count) = base();
    let y: Array1<f64> = (0..N)
        .map(|i| if (6.0 * x[i]).sin() + 0.4 * tag[i] > 0.1 * ((i % 5) as f64) { 1.0 } else { 0.0 })
        .collect();
    let fit = fit_with_design(
        bernoulli_logit(),
        Additive { terms: terms(3) },
        columns(&[&x, &tag, &count, &tag]).view(),
        y.view(),
        None,
    )
    .unwrap();
    assert_eq!(fit.aliased_coefficients(), vec![fit.beta.len() - 1]);
    assert!(fit.beta.iter().all(|b| b.is_finite()));
}

#[test]
fn two_smooths_of_one_covariate_fit() {
    // Their linear null spaces coincide; no unpenalised column can absorb it,
    // so one smooth coefficient is zeroed.
    let (x, tag, count) = base();
    let y = response(&x, &tag, &count);
    let fit = fit_with_design(
        gaussian_identity(),
        Additive {
            terms: vec![
                TermSpec::Cr { col: 0, k: 8, pc: None },
                TermSpec::Cr { col: 1, k: 8, pc: None },
            ],
        },
        columns(&[&x, &x]).view(),
        y.view(),
        None,
    )
    .unwrap();
    assert_eq!(fit.aliased_coefficients().len(), 1);
    assert!(fit.predict(columns(&[&x, &x]).view()).unwrap().iter().all(|v| v.is_finite()));
}
