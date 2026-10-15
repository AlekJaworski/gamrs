//! Parity lock for `scat` on the geometry that exposed the
//! `score_rank_adjustment` defect fixed in 0.13.0 (TF-9963): a **saturated**
//! CR basis — 5 distinct x against k = 5 — on a dollar-scale response with
//! heavy-tailed noise and very lopsided level counts, [6, 161, 432, 19, 2].
//!
//! Why this geometry: every `price_per_unit` the adjustments API reports is a
//! secant on a single-term `scat` fit made on the partial residuals of the
//! joint gaussian model, so scat sits directly under the dollar figures. With
//! the basis saturated the smooth can interpolate the five level means exactly,
//! so the REML minimum in λ is shallow — mgcv's own fixed-sp sweep (the
//! fixture's `sp_ladder`) spends only 0.04 REML units over the three decades
//! above the optimum. That makes the term a sensitive detector for anything
//! that mis-scales the penalty: with `TDist::score_rank_adjustment` returning
//! −1 the real term converged to edf 4.02 at ~30× less penalty and was 3.3%
//! off in dollars, while the three synthetic fixtures in `parity_scat.rs`
//! passed at 5e-2 throughout.
//!
//! Unlike `parity_scat_flat_ridge.rs`, this fixture HAS an interior λ optimum
//! — `saturated_basis_fixture_has_an_interior_lambda_optimum` asserts it. The
//! two fixtures catch different failure modes: a *direction/scale-of-penalty*
//! defect here, an *early-stop* defect there.
//!
//! The fixture carries mgcv's answer from all three arms mgcv can produce here
//! (`gam`+REML, `bam`+REML, `bam`+fREML). They agree with each other to 1.2e-4
//! on the fitted level, which is the point: on this term mgcv's estimator word
//! does not matter, so a single bound covers all three.
//!
//! Fixture: fully synthetic — `scripts/r/gen_scat_saturated_basis_fixture.R`,
//! seed 165. No customer data (real housing residuals stay in the gitignored
//! `data/` tree — same constraint as `docs/scat_parity_bug.md`).

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    unique_x: Vec<f64>,
    mgcv_output: MgcvArms,
    sp_ladder: Vec<Rung>,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<f64>,
    y_train: Vec<f64>,
    k: usize,
}

#[derive(Deserialize)]
struct MgcvArms {
    gam_reml: Arm,
    bam_reml: Arm,
    bam_freml: Arm,
}

#[derive(Deserialize)]
struct Arm {
    sp: f64,
    edf: f64,
    nu: f64,
    sigma: f64,
    reml: f64,
    predictions_unique_x: Vec<f64>,
}

#[derive(Deserialize)]
struct Rung {
    sp: f64,
    reml: f64,
    edf: f64,
}

fn load() -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/1d_scat_saturated_basis_n620_k5_cr.json");
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn max_rel(pred: &[f64], target: &[f64]) -> f64 {
    assert_eq!(pred.len(), target.len());
    pred.iter()
        .zip(target)
        .map(|(a, b)| (a - b).abs() / b.abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn tf9963_garage_spaces_scat_lands_on_mgcv_optimum() {
    let fx = load();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let y_var = {
        let mean = y.iter().sum::<f64>() / (n as f64);
        y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64)
    };

    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        fx.inputs.k,
    )
    .expect("gamrs::fit (scat) should not fail");
    assert!(fit.converged, "scat outer did not converge");

    let ux = Array2::from_shape_vec((fx.unique_x.len(), 1), fx.unique_x.clone()).unwrap();
    let pred = fit.predict(ux.view()).expect("predict should not fail");
    let pred = pred.as_slice().unwrap();

    let arms: [(&str, &Arm); 3] = [
        ("gam/REML", &fx.mgcv_output.gam_reml),
        ("bam/REML", &fx.mgcv_output.bam_reml),
        ("bam/fREML", &fx.mgcv_output.bam_freml),
    ];
    println!(
        "[scat saturated basis] gamrs edf = {:.4}; σ̂ = {:.1}; iters = {}",
        fit.edf_total,
        fit.scale.sqrt(),
        fit.n_iters
    );
    for (tag, arm) in arms {
        println!(
            "  vs mgcv {tag:9} (sp {:.4e}, edf {:.4}, ν {:.4}): max_rel = {:.3e}",
            arm.sp,
            arm.edf,
            arm.nu,
            max_rel(pred, &arm.predictions_unique_x)
        );
    }

    // Bar 1e-3 against each mgcv arm: observed 1.187e-4 / 1.335e-5 / 1.016e-5
    // — still closest to bam+fREML, which is the arm the engine actually runs,
    // but gam/REML is now the worst arm by 10x and so sets the bar. With
    // `score_rank_adjustment` returning −1 this fixture sits at 1.4e-2, so the
    // bound stays two orders inside where the defect lived. Left at 1e-3
    // rather than tightened: ~8x headroom on the worst arm is already the band
    // we want, and gam/REML is the arm that moves most under any outer-loop
    // change (it is the one whose λ̂ is a stopping artefact on this term).
    for (tag, arm) in arms {
        let rel = max_rel(pred, &arm.predictions_unique_x);
        assert!(
            rel < 1e-3,
            "scat vs mgcv {tag}: max_rel {rel:.3e} exceeds 1e-3"
        );
    }

    // edf is the diagnostic that actually names the failure mode: the defect
    // buys spurious degrees of freedom with too little penalty (edf 3.39 here,
    // 4.02 on the real term it was found on). Bracket rather than pin a
    // decimal — mgcv's own three arms span 2.3826-2.3879, and gamrs now sits
    // at 2.3821, i.e. just outside their span rather than 0.02 below it.
    //
    // 0.03 around the gam/REML arm is ~5x the current 0.0058 gap: it admits
    // the whole mgcv arm spread plus drift, and still rejects the 3.39 the
    // defect produced.
    let edf_gap = (fit.edf_total - 2.3879).abs();
    assert!(
        edf_gap < 0.03,
        "scat edf {:.4} is {edf_gap:.4} from mgcv gam/REML's 2.3879 — \
         the log|λS| rank convention is the first thing to check",
        fit.edf_total
    );
}

/// The fixture really has an INTERIOR λ optimum — mgcv's own score rises on
/// both sides of the sp it reports. That is what separates this fixture from
/// `parity_scat_flat_ridge.rs` (whose score descends monotonically to the
/// λ→∞ limit); if it ever stops holding, the two are testing the same regime
/// and the edf bracket above means nothing.
#[test]
fn saturated_basis_fixture_has_an_interior_lambda_optimum() {
    let fx = load();
    let at_opt = &fx.mgcv_output.gam_reml;
    let best = fx
        .sp_ladder
        .iter()
        .min_by(|a, b| a.reml.total_cmp(&b.reml))
        .expect("ladder must be non-empty");
    for r in &fx.sp_ladder {
        println!(
            "  sp={:<12.4e} reml={:>12.4} edf={:.4}",
            r.sp, r.reml, r.edf
        );
    }
    assert!(
        (best.sp - at_opt.sp).abs() / at_opt.sp < 1e-9,
        "the ladder's best rung (sp {:.4e}) is not mgcv's reported optimum (sp {:.4e})",
        best.sp,
        at_opt.sp
    );
    let (first, last) = (
        fx.sp_ladder.first().expect("non-empty"),
        fx.sp_ladder.last().expect("non-empty"),
    );
    assert!(
        first.reml > at_opt.reml && last.reml > at_opt.reml,
        "REML should be worse at both ends of the ladder ({:.4} / {:.4}) than at \
         the optimum ({:.4}) — this fixture is meant to have an interior optimum",
        first.reml,
        last.reml,
        at_opt.reml
    );
    // …and the basis really is saturated: the over-penalised end collapses to
    // the straight line, the under-penalised end spends the whole basis.
    assert!(last.edf < 2.01, "top of ladder edf {:.4}", last.edf);
    assert!(first.edf > 4.5, "bottom of ladder edf {:.4}", first.edf);
}

/// `method="fREML"` must not be a silent no-op on the scat path.
///
/// The shape-parameter driver runs damped Newton over `[ρ, log σ², log(ν−3)]`
/// and has no Fellner-Schall branch, so a fREML request used to be dropped and
/// REML/fREML came out bit-identical. The Rust API now says so; the Python
/// wrapper catches this same error, warns, and refits on REML, so callers keep
/// working while the running optimiser stops diverging from the declared one.
/// This lives beside the parity lock because scat is the family that defaulted
/// to `"fREML"` in the Python wrapper, which made every scat fit ever made
/// declare an optimiser it never ran.
#[test]
fn tf9963_scat_refuses_fellner_schall_instead_of_ignoring_it() {
    let fx = load();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());

    gamrs::outer::set_algorithm_override(gamrs::outer::OuterAlgorithm::FellnerSchall);
    let res = gamrs::fit(
        gamrs::family::tdist_identity(5.0, 1.0e9),
        x.view(),
        y.view(),
        None,
        fx.inputs.k,
    );
    gamrs::outer::clear_algorithm_override();

    let msg = match res {
        Err(e) => e.to_string(),
        Ok(_) => panic!("scat + Fellner-Schall must be refused, not silently ignored"),
    };
    assert!(
        msg.contains("fREML") && msg.contains("shape-parameter"),
        "error should name the parameter and the path; got {msg:?}"
    );
}

/// **The criterion test.** Everything else here compares where the two fitters
/// LAND. This compares the objective itself, on mgcv's own λ-slice.
///
/// The fixture's `sp_ladder` is generated with `family = scat(theta = th)` and
/// `sp = s` pinned (`scripts/r/gen_scat_saturated_basis_fixture.R:71-75`), so ν
/// and σ are held at the free fit's estimates across every rung. The ladder is
/// therefore a pure λ-slice at fixed shape, and gamrs's score can be evaluated
/// at exactly the same points.
///
/// Built on RAW `y`, so λ is in mgcv's `sp` units directly — and, it turns out,
/// the ABSOLUTE scores match too, to ~3e-6 on values of ~7325 (4e-10 relative).
/// That is asserted as well. (`FittedGam::reml_value` does NOT match absolutely,
/// because the fit core standardizes the response and the resulting offset —
/// 6471.366 here — is not the naive `n·ln(sd)` of 6492.357. That is a property
/// of the standardization, not of the criterion, which is why this test builds
/// the score directly on raw `y`.) The DIFFERENCE assertions are kept as well:
/// they cancel every λ-independent constant, so they survive any future change
/// to the scale convention.
///
/// Floor on achievable tightness: rung 3 and `gam_reml` are the same fit and
/// their recorded `reml` differs at 3.4e-11 — mgcv's own reproducibility limit.
#[test]
fn scat_criterion_matches_mgcv_on_its_own_sp_ladder() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};
    use gamrs::family::tdist_identity;
    use gamrs::inner::PirlsOpts;
    use gamrs::score::{FixedAtOneProfile, PirlsInnerBuilder, ShapeAwareEnvelopeScore};
    use gamrs::traits::{CoordsKind, ScoreDerivatives};

    let fx = load();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let m = &fx.mgcv_output.gam_reml;
    let (nu, sigma2) = (m.nu, m.sigma * m.sigma);

    let prep = Additive {
        terms: vec![TermSpec::Cr {
            col: 0,
            k: fx.inputs.k,
            pc: None,
            by: None,
        }],
    }
    .prepare(x.view())
    .expect("design");

    // Raw y, so λ is in mgcv's `sp` units directly and needs no mapping.
    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: prep.x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: prep.s_list.clone(),
        family_base: tdist_identity(nu, sigma2),
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    let v: Vec<f64> = fx
        .sp_ladder
        .iter()
        .map(|r| {
            let theta = Array1::from_vec(vec![r.sp.ln(), sigma2.ln(), (nu - 3.0).ln()]);
            score.value(&theta).expect("score at a ladder rung")
        })
        .collect();

    println!("  rung   sp            mgcv reml        gamrs score      d(mgcv)     d(gamrs)");
    let opt = 3usize; // sp = 1.3583e-6, mgcv's own optimum on this grid
    for (i, r) in fx.sp_ladder.iter().enumerate() {
        println!(
            "   {i}    {:.4e}   {:.9}   {:.9}   {:+.6}   {:+.6}",
            r.sp,
            r.reml,
            v[i],
            r.reml - fx.sp_ladder[opt].reml,
            v[i] - v[opt]
        );
    }

    // A. Ordering — no tolerance. gamrs's criterion must put its minimum on the
    //    same rung of mgcv's grid. Fires for any mis-scaled penalty.
    let argmin = (0..v.len()).min_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    assert_eq!(
        argmin, opt,
        "gamrs's criterion minimises at rung {argmin} (sp {:.4e}) where mgcv's \
         minimises at rung {opt} (sp {:.4e})",
        fx.sp_ladder[argmin].sp, fx.sp_ladder[opt].sp
    );

    // B. The steep side, three decades below the optimum. An off-by-one in the
    //    `log|λS|` rank convention — the original TF-9963 defect — shifts this
    //    by ½·ln(10³) = 3.454 units, 61% of the value.
    let want_steep = fx.sp_ladder[0].reml - fx.sp_ladder[opt].reml;
    let got_steep = v[0] - v[opt];
    assert!(
        (got_steep - want_steep).abs() < 1.0e-4,
        "steep side: gamrs {got_steep:+.6} vs mgcv {want_steep:+.6} (measured 4e-6 \
         with the observed log|H|; the working-weight log|A| gives +2.8e-2)"
    );

    // C. The shallow side, three decades above — the sharp instrument. The two
    //    candidate `log|H|`s differ by ~0.1 AND ρ-dependently, so a criterion
    //    carrying the wrong one gets B roughly right and C wrong by a large
    //    fraction.
    let want_shallow = fx.sp_ladder[6].reml - fx.sp_ladder[opt].reml;
    let got_shallow = v[6] - v[opt];
    println!(
        "  shallow: gamrs {got_shallow:+.9} vs mgcv {want_shallow:+.9}  (Δ {:+.3e})",
        got_shallow - want_shallow
    );
    assert!(
        (got_shallow - want_shallow).abs() < 1.0e-4,
        "shallow side: gamrs {got_shallow:+.9} vs mgcv {want_shallow:+.9} (measured \
         -1.9e-6 with the observed log|H|; log|A| gives -4.9e-3)"
    );

    // D. And the absolute values, rung by rung — the strongest form of the
    //    claim "gamrs's criterion IS mgcv's criterion". Measured worst 3.0e-6
    //    on scores of ~7325. Bar at 1e-3, ~300x the measurement, because this
    //    is the assertion most exposed to an unrelated convention change.
    for (i, r) in fx.sp_ladder.iter().enumerate() {
        let d = (v[i] - r.reml).abs();
        assert!(
            d < 1.0e-3,
            "rung {i} (sp {:.4e}): gamrs's ABSOLUTE score {:.9} vs mgcv's REML \
             {:.9}, off by {d:.3e}",
            r.sp,
            v[i],
            r.reml
        );
    }
}

/// **`vcov` must be the FISHER inverse** — mgcv's `Vp`, not the inverse of the
/// matrix the score differentiates.
///
/// mgcv builds `rV` from the `sqrt(wf)` re-factorisation (`gdi.c:2266-2288`) and
/// reports `Vp = rV·rVᵀ·scale`. Verified in R against mgcv 1.9.4 on this
/// fixture, using mgcv's OWN design and penalty so the bases match: the Fisher
/// inverse reproduces `m$Vp` to 7.8e-15 max relative, while the observed
/// inverse — what gamrs used to report — is 2.2e-2 out (`Vp[1,1]`
/// 1.716070e+06 vs 1.713119e+06).
///
/// Asserted here in the form that does not need mgcv's basis: for scat the
/// Fisher weight is the SCALAR `c = (ν+1)/((ν+3)σ²)`, so
/// `Vp = (c·XᵀX + λS)⁻¹` exactly. Fitted on an already-standardized response so
/// there is one scale throughout and `rescale_scat_fit` is a no-op — the raw
/// path is covered separately by
/// `entry_point_equivalence::pre_standardizing_the_response_is_a_no_op`.
#[test]
fn scat_vcov_is_the_fisher_inverse() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};
    use ndarray_linalg::Inverse;

    let fx = load();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y_raw = Array1::from_vec(fx.inputs.y_train.clone());
    let sd = {
        let m = y_raw.iter().sum::<f64>() / (n as f64);
        (y_raw.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / ((n - 1) as f64)).sqrt()
    };
    let y = y_raw.mapv(|v| v / sd);

    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, 1.0),
        x.view(),
        y.view(),
        None,
        fx.inputs.k,
    )
    .expect("scat fit");

    // c from the fit's OWN reported shape, so this cannot pass by accident.
    let sigma2 = fit.shape_params[0].exp();
    let nu = 3.0 + fit.shape_params[1].exp();
    let c = (nu + 1.0) / ((nu + 3.0) * sigma2);

    let prep = Additive {
        terms: vec![TermSpec::Cr {
            col: 0,
            k: fx.inputs.k,
            pc: None,
            by: None,
        }],
    }
    .prepare(x.view())
    .expect("design");
    let xd = &prep.x_design;
    let p = xd.ncols();
    let mut a = xd.t().dot(xd).mapv(|v| c * v);
    for (j, s_j) in prep.s_list.iter().enumerate() {
        let lambda_j = fit.rho[j].exp();
        for r in 0..p {
            for q in 0..p {
                a[[r, q]] += lambda_j * s_j[[r, q]];
            }
        }
    }
    let want = a.inv().expect("Fisher matrix should be invertible");

    let got = &fit.vcov;
    let scale = want.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let rel = got
        .iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0_f64, f64::max)
        / scale;
    println!(
        "  nu={nu:.6} sigma2={sigma2:.6} c={c:.6e}  edf={:.4}\n  \
         vcov vs (c.X'X + lam.S)^-1: max rel = {rel:.3e}",
        fit.edf_total
    );
    assert!(
        rel < 1.0e-10,
        "vcov is not the Fisher inverse (max rel {rel:.3e}). If this fires after \
         a change to compute_vcov, note that the observed inverse differs from \
         mgcv's Vp by 2.2e-2 on this fixture — the Fisher pair is the correct one."
    );
}
