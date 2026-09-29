# Changelog

All notable changes to **gamrs** are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); this project is in
beta (`0.x`), so minor bumps may carry breaking changes until the 1.0 surface
is locked. Versions correspond to the published PyPI wheels.

## [Unreleased]

### Fixed

- **A `scat` fit whose intercept came back at a fraction of the response
  level.** A 10-term `family="t-dist"` fit on ~1,800 house sales (raw prices,
  identity link) returned `coef[0]` ≈ 56k on a 200k mean — fitted values at
  0.08–0.28× the prices, `converged_ = False` after 9 outer iterations — on
  0.13.2 through 0.14.2, while 0.13.1 returned 199.9k. Gaussian was fine on
  every version.

  The defect is in the inner solve, and it is old: β̂ was solved from a second
  factorisation of `A + 1e-12·max|A_ii|·I`, a uniform ridge kept for v0.x
  byte-parity. Uniform is the problem. The `lot_sqft` term (values 1.1e3 to
  2.5e8, quantile knots leaving one interval ~1e4× longer than the rest) has a
  penalty with `max diag S = 2.5e10`; `SmartInit` pins its λ at the 1e6 cap;
  `A`'s largest diagonal is then 2.5e16 and the "rounding" ridge is 25,000 on
  an intercept diagonal of 2,147. The very first PIRLS solve returns β̂ at
  2147/(2147+25424) = 0.078 of the response level — the measured ratio to
  three digits, and it tracks `2147/(2147 + 0.0254·λ)` across λ = 1e3…1e7 —
  and PIRLS cannot leave it: every Newton step re-solves the same ridged
  system and the penalised-deviance test rejects it, 50 iterations, not
  converged. The outer Newton then optimises a criterion whose inner solution
  is not the penalised-deviance argmin, and its own REML value at where it
  stops (4028) is far above the value at 0.13.1's answer (1319).

  Why 0.13.1 got away with it: its ρ-gradient carried the spurious
  `½·c·λ·S[i*,i*]·tr(A⁻¹)` term 0.13.2 removed — proportional to λ, so on
  this term it read ~1e4 and drove λ from 1e6 to 1e-7 in the first steps,
  out of the regime where the ridge bites. Removing it was right; it exposed
  the ridge. Restoring it on head makes the fit fail at iteration 1.

  The unridged factor already had to succeed before the ridged one was ever
  built, so the ridge never guarded against a singular system. β̂ is now
  solved from the one factor the score reads
  (`linalg.rs::factor_and_solve`). On the reported fit: intercept 56,103 →
  199,853 (0.13.1: 199,869), mean(fitted)/mean(y) 0.280 → 0.999, corr 0.776
  → 0.871, converged in 45 outer iterations; the inner PIRLS at the start
  point goes from 50 iterations unconverged to 4. A synthetic two-term case
  (`tests/tdist_intercept_collapse.rs`: one covariate integer-valued in a
  narrow band with three far-out values, t₅ noise on a 200k level) goes from
  intercept 158,622 to 200,251. Not `scat`-specific — the same solve serves
  every PIRLS family and the Gaussian closed form; it needs one penalty
  block ~1e13× the intercept's diagonal to matter, which is why 24 parity
  fixtures never saw it.

- **The ρ box now ends where the factorisation stops being computable.**
  Removing the ridge above exposed a second, older defect: `A = X'WX +
  Σλ_jS_j` is factorised as is, so once `λ_j·S_j` dwarfs `X'WX` the penalty's
  null-space directions (a smooth's linear part is not axis-aligned) are
  recovered from a cancellation of order `eps·λ_j·max diag S_j / max diag
  X'X`. Past `grad_tol / eps` (≈ 4.5e8) that round-off exceeds the gradient
  tolerance and the REML surface grows a slope of its own. Measured on seed 29
  of `tests/outer_indefinite_axis.rs` at fixed shape: `g_ρ` = −6.8e-6 at
  ρ = 20, −1.1e-2 at 30, −11 at 38.6, where the true gradient decays like
  1/λ; the outer followed it to λ = 5.7e16, edf 2.63 on a curve that is a
  straight line at edf 2.0000, and reported `converged = false`. 0.14.2
  never got there by accident: its ridge began crushing the intercept around
  ρ = 25 (score 62 → 98 → 1,207 → 25,822 along ρ), a wall that stopped the
  outer at ρ = 20.

  `axis_bounds` now caps each ρ_j at `ln((grad_tol/eps)·max diag X'X / max
  diag S_j)` (still at most the old 50); the lower bound stays at −50. mgcv
  avoids the region by reparameterising (Wood 2011 Appendix B); until gamrs
  does, the box ends where the arithmetic does. The 117-fit sweep: non-
  converged 1 → 0, outer iterations 3,866 → 3,390. Seed 29 converges at
  ρ = 22.7 with REML 62.2374 (0.14.2: 62.2384) and edf 2.0000, in 86 outer
  iterations against 0.14.2's 13 — it now walks to the bound and settles ν
  there rather than stopping on the ridge's wall. Draws whose λ̂ used to sit
  above the new bound land on it and are classified converged by the
  boundary KKT test; their fits are unchanged to the printed digits because
  such a term is already at its null-space edf.

- **`scat` now starts its smoothing parameters where mgcv does.** A 10-smooth
  `family="t-dist"` fit on 1,000 house sales ended 15.3 REML units worse than
  mgcv `gam(method="REML")`: `lot_sqft` came back a straight line (edf 1.00)
  where mgcv fits a curve (edf 4.50). Both libraries score the same criterion
  (mgcv at gamrs's point: 776.164; gamrs: 776.149). Along `lot_sqft` it has
  two basins, a minimum at log λ −2.19 and, past a ridge near 6, a shelf as
  λ → ∞. `SmartInit` divides every `‖S_j‖_F` by one global `‖X‖_F²` and
  started that skewed term at 6.48; mgcv's `initial.sp` starts it at −0.11.
  mgcv's own Newton, given gamrs's start, lands on the same shelf. The basin
  boundary is between 5.00 and 5.25. With `converged_ = True` on the
  tdist-intercept branch, the wrong answer was silent.

  `scat` now uses `MgcvInit`, a port of mgcv 1.9-3's `initial.sp` on the
  design `initial.spg` weights for an extended family (`½·Dmu2` at
  `mustart`). It matches mgcv's own `initial.sp` to 1e-9 on mgcv's own `X`
  and `S`, weighted and unweighted (`tests/initial_sp_parity.rs`). On the
  heatmap benchmark's 16 fits (REML minus mgcv):
  - m04 wide goes from +15.31 to +0.00;
  - m01 wide goes from +14.32 to +3.28;
  - m02 wide goes from −8.86 to +1.24, now mgcv's basin rather than a luckier one;
  - the other 13 stay within 0.03.

  NegBin, Tweedie and ocat keep `SmartInit`. Write-up:
  `docs/scat_start_basin_bug.md`.

- **scat's default ν and σ² starts are mgcv's, and small responses are
  standardized too.** Without `df=` or `sigma2=`, a scat fit now starts at
  mgcv's `preinitialize`: ν = 3 + e^1.5 (was 5) and σ = 0.8·sd(y) (was
  sd(y)). `scat_response_scale` dropped its sd ≥ 1 floor. Under the floor, a
  response such as price/1e6 (sd ≈ 0.1) was left unstandardized, so its
  O(1)-tuned σ² start was far off and the outer loop walked it down one capped
  step at a time.

  On the heatmap benchmark:
  - m01 wide now matches mgcv (REML 16.594 vs 16.593); it was 3.3 behind with
    the new λ start alone.
  - Outer iterations fall on 12 of 16 fits (e.g. 200 → 16, 200 → 48).
  - m18 wide now hits the 200 cap.

## [0.14.2] — 2026-09-04

### Fixed

- **`scat` fits that a saturating shape parameter used to make unfittable.**
  A single-smooth `family="t-dist"` fit (k=12, n=66) that 0.13.1 fits raised
  `RuntimeError: solver did not converge after 200 iterations (last grad norm
  = 5.646e-2)`. Three separate defects in the shared outer Newton, each only
  reachable once a shape axis saturates — which for scat / ocat / tweedie is
  the normal state on data the shape parameter cannot be identified from, and
  is why 24 positive-definite parity fixtures never caught them.

  1. **A shape axis pinned at its bound throttled every axis that could still
     move.** The per-axis step cap was applied as one global shrink
     `min_i(cap_i/|s_i|)` measured over the unprojected step, so an axis
     sitting on its bound and asking for +19.4 against a cap of 1.0 shrank the
     whole step by 20x — even though that component is clamped away at every
     trial point. ρ crawled up its ridge at 0.042 per iteration instead of
     0.83, and `log σ²` moved 5e-7 while carrying a real (FD-confirmed)
     gradient of 5.6e-2. The step is now projected onto the feasible box
     before the cap is measured. On the reported fit: 200 iterations and an
     error → 29 iterations, converged, at a REML score 5.5e-5 *lower* than
     where the crawl ran out of budget — and 29 is iteration parity with
     0.13.1's 27.
  2. **A concave axis crawled to its bound.** Where curvature is negative the
     score has no interior optimum in that direction, so over the box its
     minimum is a face — a point available in closed form. Newton instead
     proposed `-g/|H|`, measured at 0.02 per iteration against a distance of
     2.0. That point is now probed directly.
  3. **Exhausting the iteration budget discarded the fit.** mgcv warns at its
     own cap (`gam.fit3.r:1656`) and returns the estimate; gamrs raised, which
     made this the only exit in the outer loop that threw — the step-failure
     exit beside it already returned a fit with a classified `converged` flag.
     Both exits now share one classifier, and `Gam.fit` emits a `UserWarning`
     when `converged_` is false so the returned-not-raised case is not silent.
     Raising the budget instead would have been the wrong lever: on the worst
     draw, iterations 89 → 220 bought 1.3e-8 REML units and $19 on a $550k
     curve, chasing ν from 1369 to 21963 on data that identifies neither.

  Measured over 117 fits of the regime (39 draws x 3 signal shapes, scat n=66
  k=12 — `tests/outer_indefinite_axis.rs`): hard failures 6 → 0, outer
  iterations 5471 → 3866 (−29%), fits needing more than 40 iterations 28 → 19.
  Inner PIRLS calls rise 9463 → 11105 on that sweep (+17%, ~9% wall) because
  the bound jump is probed on draws where it does not win.

  **What moves and what does not**, checked against the released 0.14.1
  directly rather than inferred from the test bars:

  - **Bit-identical**: eight fixtures — Gaussian smooth / near-linear /
    wiggly, Poisson, Gamma, Bernoulli, inverse Gaussian, and the well-behaved
    `1d_scat_unweighted_n300_k10` — same outer iteration counts and the same
    edf, REML and predictions to 12 significant digits. (`near_linear` is the
    flat-ridge Gaussian that 0.14.0 moved; it does not move again here.)
  - **Moves, and toward convergence**: any fit whose Hessian goes indefinite,
    which is not only `scat`. It happens wherever a curvature diagonal turns
    negative, so a Gaussian fit can reach it too — measured on the 2-D
    thin-plate smoke fixture, ρ̂ moves 2.1e-7 relative (−11.48289205 →
    −11.48289446) and the fitted values 1.1e-8, at the same 10 outer
    iterations. The new arm lands at the **lower** REML of the two, so the
    movement is toward the criterion's optimum, not away from it. A `scat` or
    `ocat` fit whose shape parameter saturates moves by much more, and that
    is the point of the release.

  Note for anyone reading a λ̂ at the edge of its box: the bound jump can
  propose exactly that (the thin-plate fit above is offered ρ = −30 on its
  first iteration) but it is a probe, accepted only on a score improvement,
  and the optimiser walks back off the clamp — that fit converges at
  ρ̂ = −11.48. A λ̂ pinned to ±30/±50 at convergence still means what it
  always meant: the ridge ran out of the box.

### Performance

`bench_scat_profile`: outer iterations 15.0 → 10.0 per fit (−33%) at unchanged
wall time (11.31 s vs 11.33 s, min of 3). `bench_nb` and `bench_multi_smooth`
neutral to slightly better. mgcv's steepest-descent trial for indefinite
problems (`gam.fit3.r:1561-1602`, enhancement (iv) of its own header) was
ported, measured at **5.2x the inner PIRLS calls** for a 2.7% iteration
reduction and no failures fixed, and declined — `Sstep = -grad/max|grad|` is
unscaled, so its largest component is whichever axis is already at its optimum.
The reasoning is recorded in `src/outer/step.rs` next to the cheap replacement.


## [0.14.1] — 2026-09-03

### Added

- **`pc` — mgcv's point constraint — now does something.** `Gam(...,
  term_pc_mapping={"x": v})` was accepted and dropped on the floor: two fits
  differing only in it returned bit-identical predictions and the same
  `edf_total_`. It now fits `s(x, pc=v)`: the smooth passes through zero at
  `x = v` and the intercept absorbs the level, exactly as
  `mgcv:::smooth.construct3` does it (the constraint row is the uncentred
  basis evaluated at the point, absorbed through the same Householder
  null-space as the centering row).

  Predictions, λ̂ and edf do **not** move — `pc` re-parameterises the same
  model space, and the CR penalty annihilates constants — so if you were
  checking `pc` by diffing predictions, that check could never have worked.
  What moves is the split between the intercept and the smooth, which is
  what a caller reading a *partial* curve (`gam[["__constant__", term]]`)
  reads. Measured against `mgcv::gam(y ~ s(gla, bs="cr", k=6) + s(conc,
  bs="cr", k=3, pc=0))` on 500 rows: the constrained term is zero at the
  point in both (5e-17 vs 4e-17), intercepts agree to 2.0e-6 relative and
  the term's curve to 7.6e-5 — the same band as the unconstrained arm of
  the same fixture (2.2e-12 / 2.3e-6).

  Available on the CR bases: `CrTerm(pc=)`, `CrStableTerm(pc=)`,
  `TermSpec::Cr { pc }`, `TermSpec::CrStable { pc }`, and
  `term_pc_mapping=` on the `predictors=` path. A `pc` that reaches a term
  with no centering constraint to replace — a parametric/linear or
  random-effect term, including a CR term demoted to parametric for having
  too few distinct values — is refused with a `UserWarning` rather than
  dropped quietly, as are `term_pc_mapping` keys that name no fitted
  predictor.

### Performance

No regression on the path that passes no `pc`, bounded at **~1%** — measured on
the new `bench_design_prepare`, which times `Additive::prepare` alone because
that is the only stage the change touches. 10 interleaved samples per arm,
separate target dirs, pinned core:

| n | master min | branch min | ratio |
|---|---|---|---|
| 500 | 0.414 ms | 0.416 ms | 1.005 |
| 2000 | 1.655 ms | 1.640 ms | 0.991 |
| 5000 | 4.093 ms | 4.096 ms | 1.001 |

`pc` itself is not a cost: free through the Python surface (0.28 ms either way
on a single smooth), and **8–9% cheaper** at design assembly for n ≥ 2000,
because the constraint replaces an O(n·k) column sum over the raw design with a
single k-vector basis evaluation.

A note for anyone benchmarking this crate: the noise floor of a whole-fit A/B on
this hardware is **~8% at n=500**, measured by running one binary against a
byte-identical copy of itself. Single-sample comparisons at that scale say
nothing.

### Known, not fixed

`design="cr_stable"` is accepted and ignored in the same way — `self.design` is
never consulted at fit time, so `coef_` comes back bit-identical to `design="cr"`
while `bs_` reports `cr_stable`. The only route to the rotated basis from Python
is `terms=[CrStableTerm(...)]`.

## [0.14.0] — 2026-08-28

**`scat` / `t-dist` numerical results change again, and by more than 0.13.0
moved them.** Six separate defects were found in the REML machinery scat
drives, each traced to a line in mgcv's source and measured against it. λ̂,
edf, fitted curves, **and every standard error** move. They move toward mgcv:
against `mgcv::gam` on a common standardized problem, the worst of ten real
smooth terms goes from a 6.0e-4 relative curve error at 0.13.1 to 3.8e-5.

If you fit `family="scat"` / `"t-dist"`, expect different numbers after
upgrading — including from `predict(se=True)`, `vcov` and any confidence
interval, which 0.13.0 did not touch.

**Other families move slightly too, and it is worth knowing why.** The
curvature, edf and vcov fixes are all behind `Loss` hooks that default to
`None`, so those are scat-only. But three of the six defects were in the
*shared* outer loop — the convergence veto, the halving budget and
`grad_tol` — so every family that reaches a flat λ ridge now walks further
along it instead of stopping where the steps went quiet. Measured across the
non-scat parity fixtures, μ agreement with mgcv gets marginally **worse**
while σ̂² agreement gets **better**:

| fixture | max_rel μ | scale_rel σ̂² | outer iters |
|---|---|---|---|
| `1d_gaussian_near_linear_n500_k10` | 6.50e-7 → 3.92e-6 | 8.97e-7 → 7.96e-7 | 10 → 12 |
| `4d_gaussian_mixed_n1000_k10` | 4.72e-6 → 3.48e-5 | 8.05e-8 → 6.48e-8 | 11 → 16 |
| `8d_neighbourhoods_like_n15000` | 5.90e-6 → 1.50e-5 | 6.10e-7 → **2.19e-7** | 9 → 11 |
| `10d_gaussian_n3000_k8` | 2.19e-6 → 6.53e-6 | 1.18e-7 → **4.13e-8** | 11 → 14 |
| `additive tw profile-p n600 k8` | 4.73e-4 → **3.30e-4** | — | — |

That is the flat-ridge effect, not a regression: on `near_linear` — a term
that is very nearly a straight line, so its ridge runs to λ→∞ — ρ̂ goes
17.770 → 19.766, further up a ridge mgcv also does not finish climbing. Every
one of these stays two or more orders inside its parity bound (the Gaussian
bars are 5e-4), and Tweedie improves. Note the **iteration counts rise
~20–45%**, so non-scat fits get somewhat slower as well.

The full write-up, including what was measured and ruled out, is
`docs/scat_adjuster_parity_2026-08.md`.

### Fixed
- **The REML score used the wrong curvature, so scat was optimising the wrong
  function.** The score's `log|H|` term was evaluated on the working-weight
  matrix `A`, but `irls_observed_pair` substitutes the positive *expected*
  curvature wherever the observed `½·D_μμ ≤ 0` — that substitution is what
  keeps `X'WX + λS` factorisable, and it is correct for the β step, but it
  means `A` is not the Hessian of the penalised deviance on those rows. mgcv
  keeps the observed weight, negatives and all (`gam.fit4.r:367`, `w <-
  dd$Deta2 * .5`), and only zeroes a row if the factorisation comes back
  indefinite.

  **This is proven against mgcv, not argued.** Evaluated on mgcv's own
  `sp_ladder` at pinned shape — a pure λ-slice, so nothing about gamrs's
  optimiser enters — gamrs's score now reproduces mgcv's REML at all 7 rungs
  to **3e-6 absolute** on values of ~7325, i.e. 4e-10 relative. The old
  working-weight `log|A|` misses the same ladder by 2.8e-2. Locked by
  `scat_criterion_matches_mgcv_on_its_own_sp_ladder`, which fails on the old
  criterion.

  One seam decides this: `family_observed_score_weights` in `pirls.rs` sets
  what `fit.a_factor` is, and `log|H|`, `tr(H⁻¹S)`, `a_inv`, the gradient's
  `h_diag` and the Hessian all read that one factor, so they cannot disagree.
  `Loss::has_observed_curvature()` lets families without a weight switch skip
  the work entirely.

- **The implicit-function-theorem bracket was the same wrong matrix.**
  `compute_rho_envelope_gradient` solved for `∂β/∂ρ` against `fit.a_factor`
  when that factor was the working-weight one. At the converged fit the
  stationarity residual is 1.4e-8, so a central FD of β̂ is a trustworthy
  oracle: `∂β/∂ρ` was **2.576e-2** against `A` and **1.988e-8** against the
  observed Hessian, and the resulting score gradient went from 9.5e-4
  relative off its own FD to 5.6e-6. The bracket is indefinite by
  construction, so it needs LU rather than Cholesky. Locked by
  `ift_dbeta_drho_matches_fd_of_beta_hat`.

- **The outer loop concluded convergence from a small score change.** mgcv
  does the opposite: `fast-REML.r:1587-1603` sets `converged <- TRUE`, clears
  it if any gradient exceeds tolerance, then clears it *again* if the REML
  value is still moving — the score-change test is a **veto**, never a reason
  to stop. gamrs returned `converged: true` on a small `|ΔREML|`, so on a flat
  λ ridge the loop stopped wherever the steps went quiet. On the synthetic
  flat-ridge fixture the criterion's argmin is at ρ ≈ 21.92 and the loop
  stopped at ρ ≈ 16.77.

- **The line search collapsed to a single probe on exactly the ridges that
  needed it most.** The step-halving cap carried a `stalled ⇒ 1 halving`
  branch ported from mgcv_rust (`smooth.rs:2741-2772`) that fired when
  `|ΔREML|/|v| < 1e-4` after iteration 3 — that is, on a flat λ ridge, where
  *more* halving is wanted. mgcv uses `maxHalf = 30` unconditionally
  (`gam.fit3.r:1230`, `mgcv.r:2212`). This was the root cause of a σ²-start
  sensitivity that had survived every other hypothesis: the saturated-basis
  fixture landed 9.1e-4 REML short from a `0.1·sd²` start and now lands
  2.5e-8 from it. Locked by `scat_fit_is_insensitive_to_the_sigma2_start`.

- **edf came from the observed pair; mgcv computes it from the Fisher pair.**
  mgcv keeps two factorisations on purpose — `gdi.c:2260-2290` finishes the
  observed-weight work, then overwrites `w <- sqrt(wf)`, re-QRs and rebuilds
  `P`, `K` and `rV` from that, "in order to compute effective degrees of
  freedom safely, and for posterior inference". For scat `wf` is the single
  scalar `(ν+1)/((ν+3)σ²)` (`efam.r:1327-1329`), so mgcv's scat edf is
  algebraically a Gaussian edf at effective smoothing `λ/c`, with no observed
  curvature in it anywhere.

  Verified in R against mgcv 1.9.4 at mgcv's own fit: mgcv reports
  `sum(edf) = 2.3879040656710795`; the Fisher-pair formula gives
  `2.3879040656711243` (**+4.5e-14**), the observed-weight pair gives
  `2.3807272805472910` (**−7.2e-3**). 23 of 620 rows had `w_obs ≤ 0`.
  New `Loss::expected_curvature_weights`, and `GaussianInnerFit` now carries a
  `fisher` pair built in PIRLS where family and penalty are both in hand.

- **`vcov` likewise, and σ² was double-counted in it.** mgcv builds `rV` from
  that same `sqrt(wf)` re-factorisation and reports `Vp = rV·rV'·scale`, so
  coefficient covariance comes from the Fisher factor too. Verified in R on
  mgcv's own design and penalty: the Fisher inverse matches `m$Vp` to
  **7.8e-15**, the observed inverse — what gamrs used — to **2.2e-2**.

  Separately, `compute_vcov` multiplied by `scale = σ²` when the Fisher weight
  `c = (ν+1)/((ν+3)σ²)` already contains `1/σ²`. That is precisely why mgcv
  reports `scale = 1` for scat: the dispersion is carried by `wf`, not by the
  multiplier. The reported vcov sat a further 31.4% off. End-to-end against
  mgcv's `Vp`, max relative difference goes **3.196e-1 → 8.7e-3**, and
  `se(β₀)` reads 1310.00 against mgcv's 1309.99. gamrs still reports
  `scale = σ²` on the `FittedGam` — a deliberate divergence — it just must not
  be applied to the Fisher inverse.

- **`Gam(sigma2=...)` and `Gam(nu=...)` were silently discarded.** Neither was
  a constructor parameter, and a `**kwargs: Any` stashed anything
  unrecognised into `self._unknown_kwargs` without a word — five orders of
  magnitude of `sigma2` gave bit-identical fits. `sigma2` is now a real
  argument forwarded to the native scat init alongside `df` (= ν); it is a
  start, not a constraint, and it is load-bearing. Unknown keywords still do
  not raise (mgcv_rust source compatibility) but now **warn**, naming the
  ignored keys.

### Added
- **`evaluate_reml_at_scat` returns the analytic shape gradient**, from the
  same score type the outer Newton drives, and reports `grad_error` on
  failure rather than omitting the key — a caller defaulting a missing `grad`
  to NaN cannot distinguish "not computed" from "computed as NaN". The scat
  shape axes are a severe cancellation (`½·d(Dp)` and `−d(ls)` at ∓~300
  summing to −8.7e-4), so nothing but the real gradient can be checked
  against an FD here.

### Changed
- **`grad_tol` is 1e-7.** Swept against mgcv 1.9.4 run directly on ten real
  smooth terms: 1e-7 is the knee of the curve *and* slightly better than
  anything tighter on both of the two worst terms (worst-term curve error
  5.9e-4 at mgcv's own 5e-6, 3.8e-5 at 1e-7, unchanged below). The curve is
  not monotone in the tolerance, so a two-point comparison misleads — the
  full sweep is in the doc comment at `outer.rs`.

- **The scat parity bounds were recalibrated to the new criterion.** They had
  been set against the old one and carried 12×–28000× headroom, wide enough
  for a real regression to pass through. They now sit at ~5–10× over the
  measured value — tight enough to bite, loose enough that last-digit drift
  across a refactor is not a false alarm. Two bars were deliberately left
  alone as already in that band. Three stale justifying comments were struck,
  including one citing a "~1e-3 residual in scat's ρ-gradient" that now
  measures 5.6e-8.

### Performance
- **scat fits cost about 39% more** (`bench_scat_profile`, n=2000 k=10:
  41 → 56 ms/fit). This is the price of the correct criterion and is
  inherent — `a_factor` is now an extra O(n·p²) weighted product plus a
  factorisation per inner fit, and mgcv does the same two factorisations.
  Two redundant rebuilds were removed on the way (the IFT no longer re-forms
  and re-factorises a matrix `fit.a_factor` already is; `score_log_det_h` no
  longer rebuilds a matrix whose log-det is `fit.log_det_a()`), and the
  Fisher pair is now opt-in per fit via `PirlsOpts::want_fisher` rather than
  built on every inner iteration and used once.

  Across n it is not uniform: n=500 1.08×, n=2000 1.17×, n=5000 1.36×,
  n=10000 1.08×, and n=2000 k=25 **0.95×** — faster.

- **Other families pay a smaller, indirect cost**: no per-iteration work was
  added for them, but the outer loop no longer stops early, so it runs more
  iterations. On the parity fixtures that is 9 → 11 at n=15000 d=8 and
  11 → 16 on `4d_gaussian_mixed`. This was not separately wall-clocked —
  `bench_gaussian` and `bench_baseline` panic on fixtures absent from the
  repo, which is a pre-existing gap, not one this release introduced.

## [0.13.2] — 2026-08-27

A packaging and correctness release: the sdist is scoped to what actually
builds and tests the extension, and scat's REML ρ-gradient is fixed.

### Fixed
- **The sdist no longer ships `docs/`, `scripts/`, `tools/` or `.github/`.**
  `[tool.maturin]` carried no `include`/`exclude`, so `maturin sdist` packaged
  the entire tracked tree. None of the excluded paths build or test the
  extension — they are documentation and local development harnesses, several
  with absolute paths that only resolve on a developer's own box. `benches/`
  stays, because `Cargo.toml` declares `[[bench]]` paths into it. Verified by
  building an sdist locally: it now contains only `benches/`, `python/`,
  `src/` and `tests/`.

- **The shape-aware REML gradient's ρ (log λ) axis differentiated a ridge that
  is not there.** `compute_rho_envelope_gradient` added
  `½·c·λ_j·S_j[i*,i*]·tr(A⁻¹)` with `c = 1e-5·(1 + √n_pen)` — the adaptive ridge
  `OcatInner` bakes into the factor it returns. `PirlsInner`, which drives
  scat/TDist, NegBin and Tweedie, hands back an **unridged** factor (its 1e-12
  ridge goes on a copy used only for the β̂ solve). For those families the term
  was pure error, and being proportional to λ it grew without bound while the
  true gradient decays like 1/λ: on the scat FD probe it read 1.1e-2 at ρ = 0
  and 9.2e4 at ρ = 16. Harmless on a steep λ ridge, decisive on a shallow one —
  it flipped the gradient's sign, the outer Newton stepped the wrong way,
  step-halving found no decrease and `outer.rs` returned the standing point.
  The Hessian is finite-differenced on this same gradient, so `H_ρρ` came back
  negative where the surface is monotone.

  Inner solvers now declare what they ridge via
  `ShapeInnerBuilder::score_ridge_scale`. Alongside it, the `log|H|` β-chain
  term prefers `Loss::ift_trace_weight_derivs`' `dw_dmu` over `½·Dmu3` where the
  family supplies it — for scat the two differ on the outlier rows, where the
  working weight is the μ-independent expected curvature.

  Measured: `tests/parity_scat_flat_ridge.rs` goes from edf 2.1316 / $291.2
  against mgcv's 2.0102 to edf 2.0015 / $22.3, and its parity assertion is no
  longer `#[ignore]`d; `3d_scat_identity_n800` tightens 4× (2.2e-3 → 5.7e-4);
  Tweedie's ρ axis goes from 2.78e-2 to 2.42e-3 relative. The ρ axis is now
  asserted for scat (`score_tests.rs::tdist_analytic_rho_grad_matches_fd`, no
  longer ignored) and for Tweedie (`tweedie_analytic_shape_grad_matches_fd` now
  probes `i in 0..3`); both previously said "g[0] is verified separately", and
  it was not. `2d_scat_identity_n600` moved the other way (5.7e-4 → 1.08e-3)
  while its ρ̂ and edf moved toward mgcv's; bound retuned to 1.5e-3 with the
  reasoning recorded at the call site. Full analysis in
  `docs/scat_adjuster_parity_2026-08.md`.

### Changed
- **`tests/parity_scat_tf9963.rs` now runs on generated data.** Its fixture is
  `tests/fixtures/1d_scat_saturated_basis_n620_k5_cr.json`, generated by
  `scripts/r/gen_scat_saturated_basis_fixture.R` (seed 165), which restates the
  geometry that made the term a detector — n = 620, k = 5, 5 distinct x with
  counts [6, 161, 432, 19, 2], dollar-scale response, heavy-tailed noise — with
  numbers that are not anyone's. mgcv's reference outputs were regenerated
  against it (all three arms). The fixture still catches the defect it was
  written for: reinstating `TDist::score_rank_adjustment() == -1` moves it to
  edf 3.39 and 1.4e-2 relative, 14× outside the test's 1e-3 bound. A new
  `saturated_basis_fixture_has_an_interior_lambda_optimum` asserts the fixture
  has not drifted into the flat-ridge regime `parity_scat_flat_ridge.rs` covers.

## [0.13.1] — 2026-08-24

Wheels only. No library code changed; `pip install gamrs` returns the same
numbers on every platform that already had a wheel.

### Added
- **macOS Intel (x86_64) wheels.** Every release up to and including 0.13.0
  shipped macOS wheels for arm64 only — `macosx_11_0_arm64`, cp311–cp314 — so
  an Intel Mac matched no wheel, fell through to the sdist, and `pip install
  gamrs` turned into a from-source build needing a Rust toolchain and a
  compile of OpenBLAS. Intel was never dropped; it was never built. The
  release matrix now has a second macOS arm building `macosx_10_13_x86_64`
  for cp39–cp314, the same interpreter range the linux wheels cover.

  It is a per-arch wheel, not a `universal2` fat one: `blas-static` has
  openblas-src compile OpenBLAS from source for the host arch, so there is no
  fat `libopenblas` to `lipo` together, and the two arches need different
  portability pins anyway (`OPENBLAS_TARGET=PRESCOTT` vs `ARMV8`). pip prefers
  an exact-arch wheel over `universal2`, so nothing is lost.

  The Intel wheel gets the same two-knob OpenBLAS portability treatment as the
  linux one — `DYNAMIC_ARCH=1` for runtime kernel dispatch, `TARGET=PRESCOTT`
  to keep the common/driver code at the x86-64 baseline — plus
  `RUSTFLAGS=-C target-cpu=x86-64`, so a newer build host cannot bake an
  instruction into code that has to run on every Mac. The deployment target is
  pinned to 10.13 rather than inherited from the build interpreter, because an
  inherited one tracks the runner's macOS version and would tag the wheel
  `macosx_15_0_x86_64` — excluding the older machines the wheel exists for.

### Fixed
- The Intel job's runner label. `macos-13` was the last Intel image on the
  `macos-N` line and was retired on 2025-12-04. A retired GitHub-hosted label
  does not fail a job, it leaves it unassigned, so the arm queued indefinitely
  instead of building. It now asks for `macos-15-intel`, which GitHub supports
  until August 2027.

## [0.13.0] — 2026-08-24

### Fixed
- **`scat` / `t-dist` numerical results change — every fit, not just edge
  cases.** `TDist::score_rank_adjustment` returned `-1`, subtracting one from
  the penalty null-space rank inside the REML score's `log|λS|₊`. That term
  contributes `−½·rank·ρ`, so dropping the rank tilted the whole REML surface
  by `+ρ/2` per smooth and the outer Newton converged **under-penalised**. The
  override arrived labelled EXPERIMENTAL on an unrelated tensor-dispatch commit
  (`fa3df55`) and sat for three months; `rank_and_log_pseudo_det` was already
  returning the count mgcv uses.

  **If you fit `family="scat"` / `"t-dist"`, expect different smoothing
  parameters, different edf and different fitted values after upgrading.** They
  move toward mgcv, not away: on the real 620-row `garage_spaces` term (5
  distinct x, k=5, saturated basis) mgcv's own fixed-`sp` REML sweep bottoms out
  at edf 2.37, and gamrs went from edf 4.018 — mgcv's own answer at ~30× less
  penalty, 0.54 REML units worse — to edf 2.398. Agreement with mgcv's fitted
  values on that term went 3.32% → 0.034%. Every scat parity fixture tightened
  8–16×, so the test bounds came down with them:

      1d_scat_unweighted_n300   4.1e-3 → 3.8e-4   bound 5e-2  → 1e-3
      scat_raw_scale_via_rust   4.1e-3 → 3.8e-4   bound 5e-2  → 1e-3
      additive_2d_scat_n600     9.1e-3 → 5.7e-4   bound 1.5e-2 → 1e-3
      additive_3d_scat_n800     1.7e-2 → 2.2e-3   bound 3e-2  → 4e-3

  `tests/parity_scat.rs` no longer says "not byte-equivalent to mgcv" — it is a
  measured 1e-3 lock. New `tests/parity_scat_tf9963.rs` locks the saturated-basis
  geometry the defect was found on (5 distinct x against k = 5) against mgcv's
  answer from all three arms it can produce (`gam`+REML, `bam`+REML,
  `bam`+fREML). The three general synthetic fixtures all passed at 5e-2
  throughout the defect's life, so that geometry is the detector. (Its fixture
  was replaced with a generated restatement in 0.13.2 — see that entry.)

### Changed
- **`method="fREML"` now fits on REML — and says so — where Fellner-Schall was
  never ported.** Fellner-Schall is a real solver (shipped in 0.6.0) and is
  still honoured on the GLM envelope driver: bernoulli, poisson, gamma,
  inverse-gaussian, quasipoisson, quasibinomial. It never reached four other
  drivers — the gaussian closed-form path, the quantile path, the profile-shape
  path (negbin), and the joint shape-parameter path (scat / tweedie / ocat) —
  and those ran damped Newton regardless, so on them `method="fREML"` was a
  **silent no-op**: REML and fREML came out bit-identical with nothing telling
  the caller.

  Those paths now emit a `UserWarning` naming the parameter, the family and the
  substitution, and fit on REML. Nothing statistical is lost and in general
  something is gained: gamrs' fREML is Fellner-Schall (Wood & Fasiolo 2017) and
  mgcv's `bam(method="fREML")` is the REML criterion computed the fast way —
  score two `sp` vectors with `sp` pinned and bam's fREML and REML criteria
  return identical numbers — so both are routes to the same criterion, and
  damped Newton on the REML score is the stronger route. Measured the same day:
  on a 620-row 7-smooth production design `bam`+fREML stalls **5.57 REML units**
  short of the converged answer and returns to it when started there, while
  gamrs' REML matches mgcv's REML per-term smoothing parameters to five
  significant figures.

  `method="fREML"` therefore keeps working everywhere it worked before; the
  only new thing a caller sees is the warning. The Rust API is the exception
  and raises `GamrsError::InvalidParameter` on those four drivers — a direct
  Rust caller is not the compatibility surface, and the Python wrapper is built
  on that error.
- **The Python constructor no longer defaults `scat`/`t-dist` to
  `method="fREML"`.** Every family now defaults to `"REML"`. This is a no-op on
  results — the fREML default was one of the silent no-ops above — but it means
  the declared optimiser is the running one, and no scat user is warned about
  an optimiser they never asked for.

### Docs
- `df=` / `nu=` for `scat` is documented as a **seed, not a constraint**, which
  is where it differs from `mgcv::scat(theta=)`: gamrs seeds the outer Newton's
  `log(ν − 3)` axis and then re-estimates it alongside λ and σ². Sweeping
  `df` 3.5 → 100 on the `garage_spaces` term leaves the fitted shape and λ
  unchanged to 4 significant figures. Pinning would need a fixed-shape-axis
  mechanism the shape-aware outer does not have for any family (negbin `theta`
  and tweedie `p` are seeds too), so this release states the behaviour rather
  than half-changing it.
- `docs/perf.md` no longer advises `method="fREML"` for gaussian at large n —
  gaussian is the closed-form path and never ran Fellner-Schall.
- README: scat parity figures updated to the post-fix bounds; the versioning
  note tracks 0.13.x.

## [0.12.3] — 2026-08-21

### Fixed
- **All-parametric designs now fit for every family, not just gaussian.** 0.12.2
  opened this path for gaussian only: the other families reach the penalised
  PIRLS solvers, and those called `combined_s`, which read the design width off
  `s_list[0]` — a penalty that does not exist when there are no smooths. So
  `fit` kept an explicit refusal for them rather than panicking in native code.

  `combined_s` now takes the design width as a parameter. Every caller already
  holds the design, so each can answer it, and an empty `s_list` assembles a
  zero penalty of the right shape instead of indexing into nothing. That turns
  penalised PIRLS into plain unpenalised IRLS, which is what an unpenalised fit
  of a non-gaussian family is — verified against closed form: a bernoulli fit of
  a saturated 2x2 design returns the log-odds to ~1e-15, its fitted values are
  the two group rates, `get_lambdas()` is empty and `edf_total` is the
  coefficient count. The special case added to `src/inner/closed_form.rs` in
  0.12.2 is gone, since the general path now covers it.

  The lasting shape is a `Penalties` type owning the list and the width together,
  so the width travels with the penalties instead of being passed alongside them
  at eleven call sites; that refactor is deliberately deferred.

## [0.12.2] — 2026-08-20

### Fixed
- **An all-parametric design is now fitted instead of refused (gaussian).** A
  predictor with only 2–3 distinct values is demoted to a parametric term, so a
  design made entirely of low-cardinality columns had no smooths at all —
  and `fit` raised *"All terms are parametric; a gamrs fit needs at least one
  smooth or random-effect term"*. That made gamrs unusable as a drop-in for
  mgcv anywhere a caller fits one submodel per feature: mgcv fits such a design
  without complaint, and any binary column (`pool`, a flag) hits it.

  With no smoothing parameters there is nothing for the outer Newton to search
  over, so `src/fit/gaussian.rs` skips it and takes the single unpenalised inner
  fit, and `src/inner/closed_form.rs` assembles a zero penalty of the design's
  own width (`combined_s` cannot infer that width from an empty `s_list`). The
  result is exactly weighted least squares: coefficients and predictions match
  `numpy.linalg.lstsq` to ~3e-7, `edf_total` equals the coefficient count,
  `scale_` is the residual variance on `n − p`, and `get_lambdas()` is empty.
  `n_iters` is 0 and `converged` is True because the answer is closed-form
  rather than iterated.

  **Gaussian only.** Every other family reaches the penalised PIRLS solvers,
  which still assume a non-empty `s_list`, so they keep an explicit refusal
  (now naming the family) rather than reaching a native panic.

## [0.12.1] — 2026-06-11

### Fixed
- **`fit_shash` robustness on near-Gaussian (and skewed / heavy-tailed) data.**
  v0.12.0 errored `SingularSystem: −Hess (Hp) is not SPD` on data with little
  skew/kurtosis (ε≈φ≈0) — common, and exactly where shash should reduce to a
  Gaussian. Two causes, both fixed in the outer REML (`src/gamlss/shash_reml.rs`):
  (1) `reml_eval`/`reml_grad` now perturb the penalised Hessian to SPD (mgcv's
  "ensure negative-definiteness" step) instead of erroring when the weakly-
  identified ε/φ directions carry ~0 curvature; (2) the smoothing-parameter
  search now clamps `ρ` to a sane range and rejects non-finite criteria — at
  absurd `ρ` the penalty `exp(ρ)·S0` overflowed the Hessian into a *finite
  garbage* `laml` the line search wrongly accepted, driving the fit to
  over-smooth μ all the way to linear (EDF→Mp). `fit_shash` now converges and
  recovers mgcv's fit on Gaussian/skewed/heavy-tailed data (e.g. Gaussian:
  μ̂ vs truth corr 0.999, EDF 12.1 vs mgcv 12.05, log-sp 4.90 vs 4.97).
  Regression guard: `fit_shash_converges_on_{gaussian,skewed,heavy_tailed}_data`
  unit tests (the earlier mgcv parity test only exercised shash-distributed
  data, so it never caught this).

### Docs
- README: added the `fit_shash` (sinh-arcsinh GAMLSS) section.

## [0.12.0] — 2026-06-11

### Added
- **GAMLSS: sinh-arcsinh (`shash`) — the first NON-orthogonal multi-linear-
  predictor family.** `gamrs.fit_shash(X, y, mu_terms=…, tau_terms=…,
  eps_terms=…, phi_terms=…)` fits all four sinh-arcsinh parameters — location
  μ(x), log-scale τ(x) (`logeb` link, σ = exp τ ≥ b), skewness ε(x), and
  log-kurtosis φ(x) — as smooths, **jointly**, to the shash likelihood. Unlike
  `gaulss`, shash's Fisher information is *not* block-diagonal, so it cannot
  alternate single-predictor fits: β for all four predictors is solved together
  by a **dense penalised block-Newton** inner solve (Levenberg-perturbed for
  ascent + step-halving), under **outer REML/LAML** smoothing-parameter
  selection driven by an **analytic gradient** (third derivatives `l3` →
  `d log|Hp|/dρ` via the implicit `dβ̂/dρ`). One fit yields every quantile via
  `ShashGamFit.predict_quantile` (the R `.shashQf` inverse-CDF); also
  `predict_eta` and `predict_params` → (μ, σ, ε, δ). Native Rust driver
  (`src/fit/shash.rs`, exported as `gamrs::fit_shash`); engine in
  `src/gamlss/{shash,shash_init,shash_inner,shash_reml}.rs`.

  Built component-by-component, each confronted with mgcv (1.9.4) **and** finite
  differences: density/grad/Hessian + the `logeb` link chain (vs mgcv `l0` and
  FD, ~1e-8); initialisation (bit-exact vs `pen.reg`); the joint inner Newton
  (recovers mgcv's intercept-only shash MLE to <1e-5, cross-checked against an
  independent BFGS optimum); `l3` and the analytic REML gradient (vs FD, ~3e-8);
  and the **outer REML fed mgcv's own designs recovers its smoothing parameters
  to ~1e-5 and EDF exactly**. End-to-end (gamrs building its OWN CR designs vs a
  2-smooth mgcv `shash` fit with `bs="cr"`): fitted η, smoothing parameters,
  total EDF and quantiles all match to **~1e-6**. (NB the reference smooths MUST
  request `bs="cr"`: mgcv's default `s(x, k)` is a *thin-plate* spline — a
  different basis — and comparing gamrs's `Cr` to it is apples-to-oranges; that
  mismatch, not any deficiency, accounted for an early ~1e-2 discrepancy seen
  during development.) Distinct from the two-stage
  `fit_quantile_lss(shape="shash")` (a per-residual MLE with a single global
  shape) — `fit_shash` is the genuine joint GAMLSS.
  Tests: `src/fit/shash.rs::fit_shash_matches_mgcv_two_smooth`,
  `tests/python/test_shash_gam.py`, `src/gamlss/*` unit suites (FD + mgcv).

### Known limitations
- `fit_shash` v1 supports at most one smooth term per predictor (one penalty
  per block); multi-smooth-per-predictor is a follow-up.
- shash, like mgcv's, needs an identifiable scale/shape — near-deterministic
  data can leave the penalised Hessian non-SPD, surfaced as a clear
  `SingularSystem` error rather than a silently bad fit.

## [0.11.9] — 2026-06-10

### Added
- **GAMLSS: Gaussian location-scale (`gaulss`) — the first multi-linear-
  predictor family.** `gamrs.fit_gaulss(X, y, mu_terms=…, sigma_terms=…)` fits
  `y ~ N(μ(x), σ(x)²)` with smooth μ(x) AND σ(x), jointly. Because the Gaussian
  location-scale Fisher information is block-diagonal (μ ⟂ log σ), the joint
  MLE is computed by **orthogonal alternating Fisher scoring** — an alternation
  of two single-predictor penalised weighted-Gaussian REML fits (μ reweighted
  by 1/σ²(x); log σ via the scale IRLS) — reusing the existing single-predictor
  fit stack rather than a dense block-Newton. One fit yields every quantile
  (`GaulssFit.predict_quantile`), monotone in τ so bands never cross. Native
  Rust driver (`src/fit/gaulss.rs`, exported as `gamrs::fit_gaulss`). Confronted
  with mgcv `gaulss`: recovers the same μ̂/σ̂ (RMSE ~3e-4 / ~1e-3) and matches
  OOS pinball to ~0.05%, ~70× faster than R `gaulss` at n=800
  (`tests/python/test_parity_multismooth.py::test_gaulss_joint_parity`,
  `src/fit/gaulss.rs` unit test). This is the seam for the broader GAMLSS class
  (`shash`/`gevlss`); non-orthogonal families will extend the alternation.
- **Distributional location-scale quantiles — `fit_quantile_lss`.** A new
  quantile path that models the whole conditional distribution (location μ(x)
  + scale σ(x), two Gaussian GAMs) and derives every quantile as
  `q_τ(x) = μ(x) + σ(x)·z_τ` from a single fit — the mgcv `gaulss`/`shash`
  view. Quantiles never cross (z_τ ↑ in τ, σ > 0) and τ is a predict-time
  argument (`QuantileLSSFit.predict_quantile(X, tau)`), so one fit serves all
  τ. `shape="gaussian"` (default, `z_τ = Φ⁻¹(τ)`, no scipy needed) or
  `shape="shash"` (fits skew/kurtosis on the standardised residuals via the
  existing SHASH helper; needed for skewed data). Pure Python over existing
  Gaussian fits — no Rust change. Confronted with mgcv `gaulss`: 2-D
  heteroskedastic OOS pinball within ~1% at τ ∈ {0.1, 0.5, 0.9}, zero
  crossings (`scripts/r/gen_quantile_lss_fixture.R`,
  `tests/python/test_parity_multismooth.py::test_additive_quantile_lss_parity`);
  the SHASH shape corrects the Gaussian shape's tail mis-coverage on skewed
  data (`test_quantile_shash.py::test_lss_shash_fixes_skewed_tails`).

## [0.11.8] — 2026-06-10

### Changed
- **Wheels now build with `panic = "unwind"`** (was `abort`). PyO3 wraps every
  binding in `catch_unwind`, so a Rust panic on a degenerate input now
  surfaces as a catchable Python `PanicException` instead of `SIGABRT`-ing the
  host interpreter. Correct failure mode for a library; negligible cost for a
  numerical crate.

### Fixed
- Penalty pseudo-determinant eigendecomposition now returns a typed
  `GamrsError::Linalg` on failure instead of `expect()`-panicking
  (`src/design/mod.rs::rank_and_log_pseudo_det`). Degenerate / rank-deficient
  designs surface as a catchable error rather than a host-interpreter abort.
- **Wheel portability: AVX-512 leak in the Linux build.** The first 0.11.8
  build tripped the SDE Haswell portability guard on an `vbroadcastsd zmm0`
  (AVX-512) instruction — the guard correctly blocked the PyPI publish (no
  broken wheel shipped). Root cause: the manylinux Rust toolchain was not
  emitting baseline x86-64 codegen. Pinned `RUSTFLAGS=-C target-cpu=x86-64`
  in the Linux release job so our Rust is deterministically baseline (and the
  sccache key can't reuse a native-arch object); the rebuilt wheel disassembles
  to zero `zmm` instructions and passes the guard. Also gated `publish-pypi` to
  release events so a candidate can be `workflow_dispatch`-tested through
  build + guard without publishing.

### Added
- **Multi-smooth (additive) Quantile/ELF.** `fit_quantile` gained a `terms=`
  argument (`CrTerm` / `TeTerm` / …) so quantile fits run on additive designs
  (`y ~ s(x0) + s(x1) + …`), not just a single smooth. The SHASH σ pilot, the
  K-fold CV σ search, and the final fit all use the additive design; σ stays a
  single family-level scale. First multi-smooth quantile parity test against
  qgam (the mgcv-family ground truth): 2-D additive OOS pinball within ±0.6%
  of qgam at τ ∈ {0.1, 0.5, 0.9}
  (`scripts/r/gen_quantile_multismooth_fixture.R`,
  `tests/python/test_parity_multismooth.py::test_additive_quantile_oos_parity`,
  `tests/quantile_smoke.rs::quantile_multismooth_additive_monotone_and_converged`).
- **Multi-smooth `scat` (scaled-t) mgcv reference parity.** 2-D and 3-D
  additive fixtures (`tests/parity_additive_scat.rs`,
  `tests/python/test_parity_multismooth.py::test_additive_scat_parity`,
  `scripts/r/gen_scat_multismooth_fixtures.R`): µ rel-err ~9e-3 (2-D) /
  ~1.7e-2 (3-D), σ̂² matching mgcv to ~0.1 %. Closes the README's
  "multi-smooth scat reference parity pending" gap.
- **First mgcv `ocat` multi-smooth parity** (`parity_ocat.rs` was smoke-only):
  `test_additive_ocat_parity` against `ocat(R=4)` on a well-posed noisy-latent
  DGP where both gamrs and mgcv converge cleanly — `predict_proba` agrees to
  ~1.8e-3 mean abs / ~98 % class agreement
  (`scripts/r/gen_ocat_multismooth_fixtures.R`).
- Synthetic, committable regression test for the v0.11.2 `scat`
  robustness-direction fix
  (`tests/parity_scat.rs::scat_downweights_high_outliers_synthetic`) — locks
  the directional contract (scat pulls *below* Gaussian under positive
  outliers) without depending on the proprietary housing fixture.
- Degenerate-input smoke test
  (`test_degenerate_input_never_aborts_the_interpreter`): NaN / inf /
  zero-variance fits raise catchable Python exceptions, never `SIGABRT`.
- `profile`-feature timers on the NegBin/Ocat profile-θ path
  (`rho_only_total`, `fit_inner_pirls`, `frozen_beta_probe`,
  `no_refresh_probe`, `hess_ift_rho`, `fit_inner_build`), matching the
  joint-Newton path's instrumentation. `cargo bench --bench bench_nb
  --features profile` now emits a per-phase breakdown; `bench_nb` gained
  realistic-size synthetic 2-D NegBin cases (n=2K, n=5K).

### Docs
- `docs/scat_parity_bug.md` rewritten as a self-contained **RESOLVED** note
  (was a stale "open" handoff referencing gitignored proprietary data).
- README: multi-smooth families line now lists `scat`; ocat paragraph documents
  the near-separable regime where mgcv itself diverges (θ≈181) or aborts while
  gamrs's θ∈(−3,3) bound stays stable.
- This `CHANGELOG.md` added.

### Internal
- `data/` (real housing sales) and `uv.lock` added to `.gitignore`.

## [0.11.7] — release the GIL during Python fits
- Fits release the GIL (`py.detach`) for the entire solve, so independent
  `Gam.fit` / `fit_quantile` calls run truly concurrently on a
  `ThreadPoolExecutor`. Set `OPENBLAS_NUM_THREADS=1` when fanning fits across a
  thread pool to avoid BLAS oversubscription.

## [0.11.6] — scat/link robustness + coverage
- scat: start `µ` at the response (mgcv `mustart <- y`); standardize the
  response in the fit core for raw-scale robustness (Rust API too).
- Links: floor `Log`/`Inverse` `d2`/`d3` denominators (no ±inf at µ≈0).
- Added additive + tensor multi-smooth extrapolation behaviour tests and
  non-identity-link extrapolation parity vs mgcv.

## [0.11.5] — wheel portability fix
- Pin `OPENBLAS_TARGET` + add an SDE AVX2 guard to fix the 0.11.4 `SIGILL` on
  AVX2-only hosts.

## [0.11.4] — scat raw-scale Rust API
- scat raw-scale standardization relocated into the fit core (Rust API now
  fits large-`y` scat correctly, not just the Python wrapper).
- CI: Node-24 / portability guard; wheel version derived from `Cargo.toml`
  (`pyproject` `dynamic`).

## [0.11.2] — scat mgcv parity (robustness-direction fix)
- **Fixed the `scat` wrong-direction robustness bug**: the IRLS outlier
  fallback paired the expected weight with the Fisher response, pulling
  outliers *toward* `y` (Gaussian-like) instead of down-weighting them. Now
  matches mgcv R to ~0.05 % on the fitted level. `min.df` 2 → 3 to match
  mgcv `scat(min.df = 3)`. See `docs/scat_parity_bug.md`.

## [0.11.1] — scat perf
- scat `0.52× → 0.77×` (vs mgcv_rust) via broadcast-expression conversions and
  batched `h_diag` matmul.

## [0.11.0] — analytic Hessian for scat + warm-start PIRLS
- Analytic Level-2 Hessian + observed-W PIRLS + warm-start for scat
  (`0.07× → ~0.52×`).

## [0.10.0] — full mgcv R outer-Newton stabilisation stack
- Ported smart θ-init from category frequencies, diagonal Hessian
  preconditioning, Gill-Murray-Wright eigen-fix, subset Newton, and the
  rank-deficient KKT convergence check. Single-smooth ocat now converges
  cleanly on every tested seed.

## [0.9.0] — predict_proba + parametric terms
- `predict_proba` for ocat; parametric (unpenalised, linear) terms baseline.

## [0.6.0] — fREML
- Single-step IRLS `fREML` (`bam()`-equivalent) — beats mgcv_rust at every
  tested scale for GLM families at large `n`.

## [0.4.0] — beta
- First beta. Multi-smooth additive, tensor products, and the ten-family
  parity battery.

[Unreleased]: https://github.com/AlekJaworski/gamrs/compare/v0.14.2...HEAD
[0.14.2]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.14.2
[0.14.1]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.14.1
[0.14.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.14.0
[0.13.2]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.13.2
[0.13.1]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.13.1
[0.13.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.13.0
[0.12.3]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.12.3
[0.12.2]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.12.2
[0.12.1]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.12.1
[0.12.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.12.0
[0.11.9]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.9
[0.11.8]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.8
[0.11.7]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.7
[0.11.6]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.6
[0.11.5]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.5
[0.11.4]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.4
[0.11.2]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.2
[0.11.1]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.1
[0.11.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.11.0
[0.10.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.10.0
[0.9.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.9.0
[0.6.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.6.0
[0.4.0]: https://github.com/AlekJaworski/gamrs/releases/tag/v0.4.0
