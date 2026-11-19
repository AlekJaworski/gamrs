//! Under Fellner-Schall (`method="efs"`) the reported `reml_value` must be
//! the REML score at the returned smoothing parameters, the same number the
//! REML path reports at those parameters.

use gamrs::design::{Additive, DesignStrategy, TermSpec};
use gamrs::family::{poisson_log, Poisson};
use gamrs::inner::{CholeskySolver, PirlsInner, PirlsOpts};
use gamrs::outer::{clear_algorithm_override, set_algorithm_override, OuterAlgorithm};
use gamrs::score::{EnvelopeScore, FixedAtOneProfile};
use gamrs::traits::ScoreDerivatives;
use ndarray::{Array1, Array2};
use std::marker::PhantomData;

#[test]
fn fellner_schall_reports_the_reml_score_at_its_lambda() {
    let n = 2000;
    let mut state: u64 = 0x5eed_f00d_1234_abcd;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut flat = Vec::with_capacity(4 * n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let c: Vec<f64> = (0..4).map(|_| next()).collect();
        let mu = (0.3 + (6.0 * c[0]).sin() + 0.8 * c[1] - 0.5 * c[2] * c[2]).exp();
        let (u, mut k, mut p) = (next(), 0.0, (-mu).exp());
        let mut cdf = p;
        while u > cdf {
            k += 1.0;
            p *= mu / k;
            cdf += p;
        }
        flat.extend_from_slice(&c);
        y.push(k);
    }
    let x = Array2::from_shape_vec((n, 4), flat).unwrap();
    let y = Array1::from_vec(y);
    let design = || Additive {
        terms: (0..4)
            .map(|col| TermSpec::Cr {
                col,
                k: 10,
                pc: None,
                by: None,
            })
            .collect(),
    };

    set_algorithm_override(OuterAlgorithm::FellnerSchall);
    let fs = gamrs::fit_with_design(poisson_log(), design(), x.view(), y.view(), None);
    clear_algorithm_override();
    let fs = fs.unwrap();
    let reml = gamrs::fit_with_design(poisson_log(), design(), x.view(), y.view(), None).unwrap();

    let prep = design().prepare(x.view()).unwrap();
    let score = EnvelopeScore::<Poisson, _, _, CholeskySolver>::with_inner(
        PirlsInner::<_, _, _, CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone().into(),
            family: poisson_log(),
            opts: PirlsOpts::default(),
            _solver: PhantomData,
        },
        Poisson,
        FixedAtOneProfile,
        y.clone(),
        prep.s_list.clone(),
        prep.rank_s_list.clone(),
        prep.mp,
        prep.log_pseudo_det_s_list.clone(),
    );
    let at_fs = score.value(&fs.rho).unwrap();
    let at_reml = score.value(&reml.rho).unwrap();

    // The oracle is the REML path's own number at its own λ, up to the inner
    // fit's convergence noise (measured 1.6e-10 relative).
    assert!((reml.reml_value - at_reml).abs() <= 1e-9 * at_reml.abs());
    assert!(fs.reml_value != 0.0, "fREML reml_value left at 0.0");
    assert!(
        (fs.reml_value - at_fs).abs() <= 1e-10 * at_fs.abs(),
        "fREML reml_value {} vs REML score at its λ {at_fs}",
        fs.reml_value
    );
    // Both optimise the same criterion; Newton lands at its minimum.
    assert!(reml.reml_value <= fs.reml_value + 1e-8 * at_fs.abs());
    assert!(fs.converged);
}
