//! `fit_bam` against mgcv `bam(method = "fREML")` on 3-D additive fixtures
//! (`scripts/r/gen_bam_freml_fixtures.R`). x2 is linear, so its λ sits on a
//! flat REML ridge where only the same Newton path lands on bam's value.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::bam::BamFamily;
use gamrs::family::{gamma_log, gaussian_identity, negbin_log, tdist_identity, Family};
use gamrs::traits::{Link, Loss, VarianceFn};
use gamrs::{fit_bam, FittedGam};

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    bam: Reference,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
    weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    sp: Vec<f64>,
    scale: f64,
    predictions_train: Vec<f64>,
    edf_per_term: Vec<f64>,
    edf_total: f64,
    reml: f64,
    iter: usize,
    irls_iter: usize,
    theta: Option<Vec<f64>>,
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// mgcv's reported θ (nb θ; scat ν, σ) from gamrs's `shape_params`.
fn theta(family: &str, fit: &FittedGam) -> Vec<f64> {
    let sp = &fit.shape_params;
    match family {
        "negbin" => vec![sp[0].exp()],
        "scat" => vec![sp[1].exp() + 3.0, (sp[0] / 2.0).exp()],
        _ => vec![],
    }
}

fn check<L, K, V>(name: &str, family: Family<L, K, V>, family_name: &str, tol: Tol)
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    Family<L, K, V>: BamFamily,
{
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(format!("tests/fixtures/{name}.json"));
    let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 3), fx.inputs.x_train.concat()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train);
    let w = fx.inputs.weights.map(Array1::from_vec);
    let terms = (0..3)
        .map(|col| TermSpec::Cr {
            col,
            k: fx.inputs.k[col],
            pc: None,
            by: None,
        })
        .collect();
    let fit = fit_bam(
        family,
        &Additive { terms },
        x.view(),
        y.view(),
        w.as_ref().map(|w| w.view()),
    )
    .unwrap();
    let r = fx.bam;

    let pred = fit
        .predict(x.view())
        .unwrap()
        .mapv(|e| fit.link_kind.inverse(e));
    let fitted = pred
        .iter()
        .zip(&r.predictions_train)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let sp = fit
        .lambda
        .iter()
        .zip(&r.sp)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let edf_term = fit
        .edf_per_term
        .iter()
        .zip(&r.edf_per_term)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let report = format!(
        "{name}: fitted {fitted:.2e} sp {sp:.2e} edf/term {edf_term:.2e} edf {:.2e} \
         reml {:.2e} scale {:.2e} iter {} vs {}",
        rel(fit.edf_total, r.edf_total),
        rel(fit.reml_value, r.reml),
        rel(fit.scale, r.scale),
        fit.n_iters,
        r.iter
    );
    let th = theta(family_name, &fit)
        .iter()
        .zip(r.theta.unwrap_or_default())
        .map(|(a, b)| rel(*a, b))
        .fold(0.0, f64::max);
    eprintln!(
        "{report} theta {th:.2e} irls {}",
        fit.stats.inner_pirls_iterations_total
    );
    assert!(fitted < tol.fitted, "{report}");
    assert!(sp < tol.sp, "{report}");
    assert!(edf_term < tol.fitted, "{report}");
    assert!(rel(fit.edf_total, r.edf_total) < tol.fitted, "{report}");
    assert!(rel(fit.reml_value, r.reml) < 1e-9, "{report}");
    assert!(rel(fit.scale, r.scale) < 1e-9, "{report}");
    assert!(th < tol.fitted, "{report} theta {th:.2e}");
    assert_eq!(fit.n_iters, r.iter, "{report}");
    if family_name != "gaussian" {
        assert_eq!(
            fit.stats.inner_pirls_iterations_total, r.irls_iter,
            "{report}"
        );
    }
}

/// Bars ~10× above the measured worst (fitted/edf/θ, then λ). The nb bars
/// cover multi-threaded BLAS: its different rounding moves the flat-ridge λ
/// by 8e-7 (fitted 8e-8), against 1e-10 with one thread.
struct Tol {
    fitted: f64,
    sp: f64,
}

const GAUSSIAN: Tol = Tol {
    fitted: 1e-9,
    sp: 1e-7,
};
const PIRLS: Tol = Tol {
    fitted: 1e-6,
    sp: 1e-5,
};

#[test]
fn bam_gaussian_flat_ridge_term() {
    check(
        "3d_gaussian_bam_freml_n2000_k10_cr",
        gaussian_identity(),
        "gaussian",
        GAUSSIAN,
    );
}

#[test]
fn bam_gaussian_prior_weights() {
    check(
        "3d_gaussian_weighted_bam_freml_n2000_k10_cr",
        gaussian_identity(),
        "gaussian",
        GAUSSIAN,
    );
}

#[test]
fn bam_negbin_estimated_theta() {
    check(
        "3d_nb_log_bam_freml_n2000_k10_cr",
        negbin_log(1.0),
        "negbin",
        PIRLS,
    );
}

#[test]
fn bam_scat_prior_weights() {
    check(
        "3d_scat_identity_weighted_bam_freml_n1500_k10_cr",
        tdist_identity(5.0, 1.0),
        "scat",
        PIRLS,
    );
}

#[test]
fn bam_gamma_estimated_scale() {
    check(
        "3d_gamma_log_bam_freml_n2000_k10_cr",
        gamma_log(),
        "gamma",
        PIRLS,
    );
}
