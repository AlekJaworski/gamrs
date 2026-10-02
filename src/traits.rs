//! Core trait skeletons (per v2 plan §3).

use ndarray::{Array1, Array2, ArrayView2};

use crate::Result;

/// Layer 1 — a basis defines what β means: design matrix, derivatives,
/// and penalty matrices in coefficient space.
///
/// `x` is always `(n_obs, n_input_dims)` so tensor products `te(s(x), s(z))`
/// can be added later without changing the trait shape. Phase 0 univariate
/// bases pass `n_input_dims = 1` and ignore the `axis` parameter on `d1`.
pub trait Basis {
    /// Number of basis functions = length of β when fitted on this basis.
    fn dim(&self) -> usize;

    /// Number of input dimensions the basis consumes. Phase 0 always 1.
    fn input_dim(&self) -> usize;

    /// Design matrix rows for evaluation points. `x.shape() = (n_obs,
    /// input_dim())`. Returns `(n_obs, self.dim())`.
    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64>;

    /// `∂design/∂x_axis` evaluated at `x`. Same shape as `evaluate`.
    /// `axis` is the input dimension to differentiate against. For a
    /// univariate basis pass `axis = 0`.
    fn d1(&self, x: ArrayView2<f64>, axis: usize) -> Array2<f64>;

    /// One penalty matrix per smoothing parameter slot. CR spline returns
    /// exactly one; tensor products will return one per marginal.
    fn penalties(&self) -> Vec<Array2<f64>>;
}

/// Layer 1.5 — a linear constraint/rotation on top of an inner basis.
/// Default `Basis` impl is provided via blanket impl downstream.
pub trait BasisTransform {
    type Inner: Basis;
    fn inner(&self) -> &Self::Inner;
    /// Constraint matrix `C` of shape `(k_inner, k_self)`. New basis is
    /// `B_self = B_inner · C`, new penalties are `C' · S · C`.
    fn matrix(&self) -> ArrayView2<'_, f64>;
}

/// Per-observation Level-1 shape derivatives — the family-specific
/// payload the shape-aware envelope score consumes when computing its
/// analytic θ-gradient (mgcv's `reml_grad_ocat_theta_block_analytic`
/// recipe, generalised). Returned by `Loss::level1_shape_derivatives`.
///
/// Shapes:
/// - `dmu3`: `(n,)` — `∂³D / ∂μ³`.
/// - `dth`:  `(n, n_θ)` — `∂D / ∂θ_k`.
/// - `dmuth`: `(n, n_θ)` — `∂(∂D/∂μ) / ∂θ_k`.
/// - `dmu2th`: `(n, n_θ)` — `∂(∂²D/∂μ²) / ∂θ_k`.
#[derive(Clone)]
pub struct Level1ShapeDerivs {
    pub dmu3: ndarray::Array1<f64>,
    pub dth: ndarray::Array2<f64>,
    pub dmuth: ndarray::Array2<f64>,
    pub dmu2th: ndarray::Array2<f64>,
}

/// Per-observation Level-2 shape derivatives — the second-order analogue
/// of [`Level1ShapeDerivs`], consumed by the full analytic-Hessian path
/// (port of mgcv_rust `reml/mod.rs::tdist_gdi2_native`'s `gdi2`-style
/// assembly). Returned by `Loss::level2_shape_derivatives`.
///
/// The shape-pair packing convention is **upper-triangular row-major**:
/// for `n_θ` shape parameters, `n_pairs = n_θ·(n_θ+1)/2` and
/// `pair_index(i, j) = i·n_θ - i·(i-1)/2 + (j - i)` with `i ≤ j`.
/// For `n_θ = 2`: `(0,0) → 0`, `(0,1) → 1`, `(1,1) → 2` — matching the
/// `dth2`/`det_th2`/`det2_th2` layout in mgcv_rust's `tdist_dd_arrays`.
///
/// Shapes:
/// - `dmu4`: `(n,)` — `∂⁴D / ∂μ⁴`.
/// - `dmu3_th`: `(n, n_θ)` — `∂(∂³D/∂μ³) / ∂θ_k = ∂⁴D / (∂μ³ ∂θ_k)`.
/// - `dth2`: `(n, n_pairs)` — `∂²D / (∂θ_i ∂θ_j)` (i ≤ j).
/// - `dmu_th2`: `(n, n_pairs)` — `∂³D / (∂μ ∂θ_i ∂θ_j)` (i ≤ j).
/// - `dmu2_th2`: `(n, n_pairs)` — `∂⁴D / (∂μ² ∂θ_i ∂θ_j)` (i ≤ j).
///
/// Prior weights are baked into every array per the
/// [`Level1ShapeDerivs`] convention (mgcv's `efam.r:2814-2832`).
#[derive(Clone)]
pub struct Level2ShapeDerivs {
    pub dmu4: ndarray::Array1<f64>,
    pub dmu3_th: ndarray::Array2<f64>,
    pub dth2: ndarray::Array2<f64>,
    pub dmu_th2: ndarray::Array2<f64>,
    pub dmu2_th2: ndarray::Array2<f64>,
}

/// Helper for the upper-triangular shape-pair packing used by
/// [`Level2ShapeDerivs`]. Returns the column index in the `dth2` /
/// `dmu_th2` / `dmu2_th2` arrays for the pair `(i, j)` with `i ≤ j`.
#[inline]
pub fn shape_pair_index(i: usize, j: usize, n_theta: usize) -> usize {
    debug_assert!(i <= j && j < n_theta);
    i * n_theta - i * (i.saturating_sub(1)) / 2 + (j - i)
}

/// Layer 2a — `D(y, μ)` per observation, plus first/second derivatives in
/// μ. PIRLS uses `d_loss_dmu` / `d2_loss_dmu` to build working weights and
/// working response; the score body uses `deviance_per_obs` and
/// `saturated_log_lik`.
pub trait Loss {
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64;
    /// Saturated log-likelihood — required for REML/LAML criterion.
    ///
    /// `scale` is the dispersion σ² at the current outer probe (the σ²
    /// the score body uses in its `Dp/(2σ²) + (n-Mp)/2·log(2π·σ²)` term).
    /// For families whose saturated log-lik is genuinely y-only and
    /// scale-free (Gaussian/Bernoulli/Poisson at canonical units) the
    /// argument is ignored. Gamma, InverseGaussian, scat/TDist and
    /// Tweedie all depend on σ² via lgamma / log terms; those impls
    /// MUST read it (v0.2 port, 2026-05-24).
    fn saturated_log_lik(&self, y: f64, scale: f64) -> f64;
    /// `∂D/∂μ` per observation. Used inside the PIRLS working response.
    /// Default returns `2(y - μ)` for the Gaussian squared-error case.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        -2.0 * (y - mu)
    }
    /// `∂²D/∂μ²` per observation. Default `2.0` for Gaussian.
    fn d2_loss_dmu(&self, _y: f64, _mu: f64) -> f64 {
        2.0
    }
    /// Known dispersion if any; `None` means "profile σ² from the fit"
    /// (Gaussian, Gamma); `Some(1.0)` means "σ² is fixed at 1" (Bernoulli,
    /// Poisson). This is what lets `ClosedFormEnvelopeScore` dispatch
    /// between the profiled-σ² and known-σ² REML formulae.
    fn fixed_dispersion(&self) -> Option<f64> {
        None
    }

    /// Profile σ̂²(ρ) that solves `∂REML/∂σ² = 0` at fixed ρ — i.e. the
    /// σ² that the REML score formula plugs in for THIS family's
    /// saturated log-lik. Default is mgcv's Gaussian closed form
    /// `Dp/(n - Mp)`, which is exact for any family whose sat_lik has
    /// the form `-n/2·log(2πφ) + (y-only piece)` (Gaussian, InverseGaussian,
    /// QuasiPoisson, QuasiBinomial). Gamma overrides — its sat_lik
    /// `k(φ) = -lgamma(1/φ) - log(φ)/φ - 1/φ` makes the profile equation
    /// `F(φ) = dp + 2n[ψ(1/φ) + log φ] + Mp·φ = 0` which needs Newton
    /// (see `src/pirls/mod.rs::estimate_phi_mgcv`).
    ///
    /// `phi_init` is a warm-start hint (typically the previous outer
    /// iteration's φ̂).
    fn profile_score_sigma2(&self, dp: f64, _n_obs: usize, n_minus_mp: f64, _phi_init: f64) -> f64 {
        (dp / n_minus_mp.max(1.0)).max(1e-8)
    }

    /// Whether `profile_score_sigma2` is the closed-form `Dp/(n−Mp)`.
    /// `true` for every family that does NOT override `profile_score_sigma2`
    /// (Gaussian, InverseGaussian, QuasiPoisson, QuasiBinomial). Gamma —
    /// whose profile σ̂² is the root of a Newton-on-φ equation — overrides
    /// this to `false`.
    ///
    /// The analytic outer-Newton Hessian uses this to decide which closed-
    /// form to use for `∂σ²/∂ρ_i`: closed-form profiles use the cheap
    /// `λ_i·β'S_iβ/(n−Mp)`, Newton-on-φ profiles defer to
    /// `profile_sigma2_drho_factor` for the implicit-function-theorem chain.
    fn profile_sigma2_is_closed_form(&self) -> bool {
        true
    }

    /// Implicit-function-theorem factor `−1 / F'(φ̂)` for the Newton-on-φ
    /// profile families (Gamma). Returns `Some(f)` so the score body can
    /// build `∂σ²/∂ρ_i = f · ∂dp/∂ρ_i = f · λ_i·β'S_iβ`; `None` means
    /// "no analytic chain available, caller treats σ² as constant in the
    /// Hessian (matches mgcv_rust default-OFF `MGCV_SIGMA_CHAIN`)".
    ///
    /// For Gamma: `F(φ) = dp + 2n[ψ(1/φ) + log φ] + Mp·φ = 0`, so
    /// `F'(φ̂) = (2n/φ̂)[1 - ψ'(1/φ̂)/φ̂] + Mp` (the exact same `fp` the
    /// `profile_score_sigma2` Newton iter uses for `delta = -f/fp`). This
    /// is mathematically the right chain term; mgcv_rust drops it in the
    /// Hessian (relies on it vanishing at σ²̂'s stationary point), gamrs
    /// can do better with the cheap exact value.
    ///
    /// Default `None` — closed-form profiles handle the chain via the
    /// closed-form path in `MgcvTwoSigmaProfile::dispersion_drho`.
    fn profile_sigma2_drho_factor(&self, _sigma2: f64, _n_obs: usize, _mp: usize) -> Option<f64> {
        None
    }

    /// Whether the analytic Hessian should skip the `∂W/∂η` chain term
    /// for this family. Default `false` — `EnvelopeScore::hess_analytic`
    /// includes the W-chain whenever `dw_deta` is available (Bernoulli,
    /// Poisson, InverseGaussian, etc.).
    ///
    /// Gamma overrides to `true`: mgcv_rust uses `reml_hessian_mgcv_exact_
    /// closed_form` for Gamma (`nn_exploring/src/smooth.rs:48-53` —
    /// "IFT differentiates the true GLM deviance, while the line search
    /// still evaluates working-response REML. Keep those paired
    /// derivatives on the consistent closed-form path for the two parity-
    /// sensitive edge cases"). That formula carries only the trace curvature
    /// (`tr(A⁻¹ S_i A⁻¹ S_j)`) + `bSb·A⁻¹S` data-fit pieces with no W-chain.
    /// Without this opt-out, gamrs's PIRLS-populated `dw_deta` engages the
    /// W-chain (designed for InverseGaussian+log) which adds noise the
    /// Newton-on-φ path doesn't pair with — analytic vs FD Hessian disagree
    /// by ~7.7e-4 instead of the canonical ~1e-10.
    fn skip_w_chain_in_hessian(&self) -> bool {
        false
    }

    /// Whether the analytic envelope Hessian may drive the outer Newton when
    /// the gradient carries the Newton-weight β-chain (Tk·KK'). Default
    /// `false`: that chain is not differentiated analytically, so the outer
    /// loop takes an FD Hessian of the exact gradient. The stopping point is
    /// set by the gradient either way; the Hessian only shapes the steps.
    ///
    /// Gamma opts in. Measured on three 2-smooth fits (CPU time, single
    /// thread): log link 42 ms analytic vs 140 ms FD, inverse link 73 vs
    /// 230 ms, same iteration counts and the same distance to mgcv.
    /// InverseGaussian does not: its analytic Hessian carries a Fisher-W
    /// chain that does not pair with the Newton-W gradient, and it took 6-7
    /// outer iterations instead of 4-5.
    fn analytic_hessian_with_newton_chain(&self) -> bool {
        false
    }

    /// Number of family-shape parameters this loss owns. Zero for
    /// Gaussian/Bernoulli/Poisson. TDist returns 2 (`log σ²`, `log(ν-2)`).
    /// Tweedie returns 2 (`log φ`, `p_transform`). Used by the outer Newton
    /// to know how to size θ.
    fn n_shape_params(&self) -> usize {
        0
    }
    /// Update this loss's shape parameters from a transformed θ-slice.
    /// The slice has length `n_shape_params()`. Default no-op for stateless
    /// losses. Each Loss documents its own transform (typically `log(·)`
    /// for positive scale params, `log(· - lower_bound)` for bounded ones).
    fn set_shape_params(&mut self, _params: &[f64]) {}
    /// Read the current shape parameters back as the transformed θ-slice.
    /// Inverse of `set_shape_params`. Useful for seeding the outer Newton
    /// from a fitted state.
    fn get_shape_params(&self) -> Vec<f64> {
        Vec::new()
    }

    /// Per-shape-axis step cap for the outer Newton (mgcv-style). Length
    /// `n_shape_params()`. Default `0.5` per axis — conservative enough
    /// for the typical log-space shape transforms (ocat θ, NegBin log θ).
    /// Families with looser caps in mgcv override: TDist log σ² = 1.0;
    /// Tweedie log φ = 1.0, p-transform = 2.0.
    fn shape_axis_step_caps(&self) -> Vec<f64> {
        vec![0.5; self.n_shape_params()]
    }
    /// Per-shape-axis (lo, hi) bounds clamped after each accepted Newton
    /// step. Length `n_shape_params()`. Default `(-10.0, 10.0)` — covers
    /// every gamrs Loss's transform domain (log of positive scales, log
    /// gaps for ocat). Families with tighter clamps override.
    fn shape_axis_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0); self.n_shape_params()]
    }

    /// Optional Level-1 shape derivatives — used by the shape-aware
    /// envelope score's analytic θ-gradient path. Returns `None` to opt
    /// out (score falls back to FD on score values). When `Some`, the
    /// score uses the IFT-based gradient assembly from mgcv's
    /// `reml_grad_ocat_theta_block_analytic` (ocat) and equivalents.
    ///
    /// Convention: all four arrays exclude prior-weights — caller multiplies
    /// `wt[i]` into the per-θ assembly step (mgcv efam.r:2814-2832).
    fn level1_shape_derivatives(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<Level1ShapeDerivs> {
        None
    }

    /// Per-row ingredients for the IFT trace term `tr(H⁻¹ ∂H/∂θ_k)` when
    /// the family's `irls_observed_pair` uses a per-row observed → expected
    /// (Fisher) weight switch. The score's `log|H|` factorises
    /// `A = X' diag(W) X + λS` with `W` the **actual** working weight: the
    /// observed curvature `½·d²D/dμ²` on core rows and the *expected*
    /// curvature `½·E[d²D/dμ²]` on outlier rows. The generic trace term
    /// differentiates `½·d²D/dμ²` everywhere, which is wrong on the outlier
    /// rows (where `W` is the expected curvature — a different function of
    /// `θ`, μ-independent). For scat this stalled the outer Newton in the
    /// ν → 2 corner (`data/SCAT_PARITY_BUG.md`).
    ///
    /// Returns, in the family's outer-θ convention, the **actual** A-weight
    /// derivatives so the trace term differentiates the same `A` the score
    /// factorises:
    ///   - `dw_dtheta[i, k] = ∂W_i/∂θ_k` at fixed μ,
    ///   - `dw_dmu[i]       = ∂W_i/∂μ_i` (the μ-chain coefficient; the
    ///     caller forms `dw_dmu[i] · (X·dβ/dθ_k)_i`).
    /// Both already include the per-row prior weight (same convention as
    /// [`Level1ShapeDerivs`]) and are in working-weight coordinates
    /// (identity-link assumed, as for scat). `None` ⇒ the generic
    /// `½·(Deta2th + Deta3·x_db)` path applies (NegBin/IG/Tweedie).
    fn ift_trace_weight_derivs(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<(ndarray::Array2<f64>, ndarray::Array1<f64>)> {
        None
    }

    /// `∂r_i/∂θ_k` for the score-equation residual `r` with
    /// `∇_β(½D) = −X'r`, in the outer-θ convention. Lets the shape gradient
    /// carry `log|H|`'s β-chain through dβ̂/dθ = H_obs⁻¹ X' ∂r/∂θ. `None`
    /// where the family's shape gradient does not need it.
    fn shape_score_residual_derivs(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<ndarray::Array2<f64>> {
        None
    }

    /// Per-row **observed** curvature `½·d²D/dμ²`, on every row, including
    /// where it is negative.
    ///
    /// This is NOT the working weight `A` is built from. Families with a
    /// per-row observed → expected switch (`irls_observed_pair`) substitute
    /// the positive expected curvature wherever observed ≤ 0, so that
    /// `X'WX + λS` stays factorisable. That substitution preserves β̂ — the
    /// working response is built to keep `W(z − η) = −½·D_μ` regardless of
    /// the weight, so the fixed point is still `λSβ = X'(−½·D_μ)` — but it
    /// means `A` is **not** the Hessian of the penalised deviance on those
    /// rows.
    ///
    /// The implicit-function derivative is where that bites. Differentiating
    /// the stationarity condition gives
    /// `∂β/∂ρ_j = −λ_j·[X'diag(½·D_μμ)X + Σλ_kS_k]⁻¹·S_j·β`, and the bracket
    /// is the observed Hessian — not `A`. Solving against `A` instead is a
    /// measured 2.6% error in `∂β/∂ρ` on the scat FD probe at ρ = 0 (stable
    /// across `h`), against 2e-8 when solved against the observed Hessian.
    /// mgcv does not have this problem because `gam.fit4.r:367` keeps the
    /// observed weight (`w <- dd$Deta2 * .5`, negatives and all) and only
    /// zeroes rows if the factorisation comes back indefinite.
    ///
    /// Includes the per-row prior weight, and is in working-weight
    /// coordinates (identity link assumed, as for scat). `None` ⇒ the family
    /// has no observed/expected switch, so `A` already *is* the observed
    /// penalised Hessian and the IFT can reuse its factor.
    fn observed_curvature_weights(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<ndarray::Array1<f64>> {
        None
    }

    /// Per-row **expected** (Fisher) curvature `½·E[d²D/dμ²]` — the weight
    /// mgcv computes **edf and the coefficient covariance** from, and NOT the
    /// weight it takes `log|H|` off.
    ///
    /// mgcv keeps two factorisations on purpose. `gdi.c:2260-2290` finishes the
    /// observed-weight work, then overwrites `w ← sqrt(wf)`, re-QRs
    /// `[sqrt(wf)X; E]` and rebuilds `P`, `K`, `rV` from *that*, with the
    /// comment "get rV and K using E(W)… needed in order to compute effective
    /// degrees of freedom safely, and for posterior inference".
    /// `wf = pmax(0, dd$EDeta2 * .5)` (`gam.fit4.r:563`), and for scat
    /// `efam.r:1327-1329` makes `EDmu2` the constant `2(ν+1)/((ν+3)σ²)`, so
    /// `wf = (ν+1)/((ν+3)σ²)` — one scalar, identical on every row.
    ///
    /// So mgcv's scat edf is `tr[(c·X'X + λS)⁻¹·c·X'X]`, algebraically a
    /// *Gaussian* edf at effective smoothing `λ/c`. Verified against mgcv
    /// 1.9.4 on `1d_scat_saturated_basis_n620_k5_cr`: the Fisher pair
    /// reproduces mgcv's reported `edf = 2.3879040656710795` to **4.5e-14**,
    /// while pairing the observed weights with the observed factor gives
    /// 2.3807272805 — off by 7.2e-3, because 23 of 620 rows have
    /// `½·D_μμ ≤ 0`.
    ///
    /// This deliberately breaks the `a_factor`/`a_weights` pairing invariant:
    /// that pairing is right for `log|H|` (both observed) and exactly wrong for
    /// edf (both Fisher). One factor cannot serve both jobs, which is why mgcv
    /// builds two.
    ///
    /// Includes the per-row prior weight; identity-link coordinates. `None` ⇒
    /// the family has no separate expected curvature, so edf keeps using the
    /// factor the fit already built.
    /// Does this family have a separate **observed** curvature, i.e. does
    /// [`Self::observed_curvature_weights`] return `Some`? A cheap predicate so
    /// callers can skip building `s_total` for a family that has none, without
    /// allocating a weight vector to find out.
    fn has_observed_curvature(&self) -> bool {
        false
    }

    fn expected_curvature_weights(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<ndarray::Array1<f64>> {
        None
    }

    /// Per-observation Level-2 shape derivatives feeding the **full
    /// analytic** REML/LAML Hessian path (port of mgcv_rust
    /// `src/reml/mod.rs::tdist_gdi2_native`'s `gdi2`-style assembly,
    /// itself a port of mgcv's C `gdi2`).
    ///
    /// Pre-condition for shipping: any family that overrides
    /// `level2_shape_derivatives` MUST also override
    /// [`Self::level1_shape_derivatives`] (Level-2 is meaningful only on
    /// top of Level-1's `(dmu3, dth, dmuth, dmu2th)` chain). The
    /// shape-aware Hessian dispatch checks both — if Level-2 is `Some`,
    /// it runs the closed-form joint Hessian; otherwise it falls back to
    /// central FD on the Level-1 IFT gradient.
    ///
    /// Convention: every array is **post-Jacobian** in the family's
    /// outer parameter convention (e.g. TDist returns derivatives w.r.t.
    /// `θ = [log σ², log(ν − 2)]`, NOT mgcv's native `[log(ν − 2), log σ]`).
    /// Prior weights are baked in per the [`Level1ShapeDerivs`] contract.
    fn level2_shape_derivatives(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<Level2ShapeDerivs> {
        None
    }

    /// `Σᵢ ∂²ls(y_i; scale) / (∂θ_i ∂θ_j)` packed upper-triangular per
    /// [`shape_pair_index`]. The full analytic Hessian path subtracts
    /// this per axis pair (the `−ls2` row of mgcv `gam.fit5.r:1668`).
    ///
    /// Default returns `vec![0.0; n_θ·(n_θ+1)/2]` — correct when
    /// `saturated_log_lik` is θ-independent (Bernoulli/Poisson/Gaussian/
    /// Ocat). Families with θ-dependent ls (TDist, NegBin, Tweedie via
    /// φ-Hessian) MUST override or the Hessian's `∂²ls/∂θ²` term is
    /// silently zeroed.
    fn sum_saturated_log_lik_d2theta(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _scale: f64,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Vec<f64> {
        let n_theta = self.n_shape_params();
        vec![0.0; n_theta * (n_theta + 1) / 2]
    }

    /// `Σᵢ ∂ls(y_i; scale) / ∂θ_k` for k in 0..n_shape_params — the
    /// shape-axis derivatives of the **total** saturated log-likelihood,
    /// summed across observations. The IFT analytic shape gradient at
    /// `shape_aware/gradient.rs:343` subtracts this per shape axis (mgcv
    /// `gam.fit5.r:1668`, the `-ls$d1` row of the shape block).
    ///
    /// Default returns `vec![0.0; n_shape]` — correct for any family whose
    /// `saturated_log_lik` is θ-independent (ocat: ls≡0;
    /// Bernoulli/Poisson/Gaussian: ls depends only on y/φ). Families with
    /// θ-dependent ls (NegBin, scat/TDist, Tweedie shape derivatives) MUST
    /// override.
    ///
    /// Prior weights are NOT applied here — the caller multiplies in
    /// `wt[i]` at the assembly step.
    fn sum_saturated_log_lik_dtheta(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _scale: f64,
        _prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Vec<f64> {
        vec![0.0; self.n_shape_params()]
    }

    /// Per-family rank adjustment applied at the score-formula's
    /// `Σ rank·log λ` term. Default 0 (use the mathematically-correct
    /// positive-eigenvalue count). Ocat returns −1 to match v0.x's mgcv
    /// `non_zero_rows − 2` heuristic for centered CR splines — closes the
    /// 1.23-unit score-formula offset that was driving 5.95% multi-smooth
    /// ocat μ-RMSE (parity diagnostic 2026-05-28).
    ///
    /// **Why family-scoped, not basis-scoped**: empirically, applying this
    /// adjustment globally regresses Gaussian additive and other shape-aware
    /// families that had been at their existing "documented parity floor"
    /// thanks to a different error cancellation. The mgcv heuristic is
    /// mathematically off-by-one for centered CR but downstream code
    /// (PIRLS, σ̂², etc.) is wired around that convention in family-specific
    /// ways. Ocat is the only family whose convergence basin was visibly
    /// driven by this; others stay on `0` until per-family investigation.
    fn score_rank_adjustment(&self) -> i32 {
        0
    }

    /// Per-family inner-PIRLS β-change convergence tolerance. Default
    /// `1e-9` (gamrs's historical PirlsOpts default). Families that need
    /// to match v0.x's per-family inner-PIRLS tolerance override:
    /// ocat returns `1e-8` matching `fit_pirls_ocat`'s `tolerance` arg
    /// passed from `lib.rs:968`.
    ///
    /// The hook lets the shape-aware driver pass v0.x's per-family
    /// tolerance into `PirlsOpts.dev_rel_tol` at score construction —
    /// shrinks the Layer-3 β-residual to v0.x's stopping point.
    fn pirls_dev_rel_tol(&self) -> f64 {
        1.0e-9
    }

    /// Force the joint-Newton outer to use full FD on the REML score
    /// VALUE (instead of the analytic-IFT/FD-on-grad path) when computing
    /// the joint Hessian. Default `false` — families with reliable
    /// analytic shape derivatives stay on the cheaper IFT path.
    ///
    /// **Ocat returns `true`**: the ordered-thresholds + η surface has a
    /// near-flat coordinated-shift ridge that the partial-analytic Hessian
    /// captures poorly (FD-on-analytic-grad along the shape axes leaves
    /// the ρ-θ off-diagonal information sparse). mgcv_rust uses full FD
    /// on the score value for ocat exactly for this reason — see
    /// `~/vibe_coding/nn_exploring/src/smooth.rs:622` (`reml_joint_ocat_finite_diff`).
    ///
    /// Cost: `1 + 2·d²` PIRLS solves per outer iter (d = n_terms +
    /// n_shape) vs `1 + 2·n_shape` for the IFT path. For ocat at multi-
    /// smooth with R=4 and 2 smooths, d=4 → 33 PIRLS/iter vs 5. Slower
    /// per iter, but the joint Hessian captures cross-coupling that
    /// stabilises the outer Newton on the scale-indeterminacy ridge.
    fn prefers_full_fd_hessian(&self) -> bool {
        false
    }

    /// Analytic contribution to the score's gradient w.r.t. THIS loss's
    /// shape parameters, evaluated at the current shape values.
    ///
    /// Returns `Some(grad)` of length `n_shape_params()` if the family has
    /// analytical derivatives for ALL its shape-related score terms (i.e.
    /// the caller can use the returned vector directly as the
    /// `g[1 + k]` slice for shape param k); `None` means the caller should
    /// fall back to FD on the score-value (current Phase-2 default).
    ///
    /// Inputs (all evaluated at the converged PIRLS β̂ for the current
    /// (ρ, shape) probe):
    /// - `y` — response.
    /// - `mu` — fitted μ̂.
    /// - `dp` — `D + λβ'Sβ` at converged β̂ (the "penalised deviance").
    /// - `n_minus_mp` — `n − Mp`, the score formula's denominator coefficient.
    /// - `phi_score` — the score's current σ² (`Profile` output, or
    ///   `fixed_dispersion()` for shape-managed dispersion families like
    ///   Tweedie). The impl uses this for the `D/(2φ)` and `log(2πφ)` terms.
    ///
    /// Envelope-theorem assumption: at PIRLS convergence, ∂(β̂)/∂(shape) does
    /// NOT contribute to the score gradient — only the explicit
    /// shape-dependence does. This matches v0.x's `RemlScoreParts` assembly.
    ///
    /// Phase-1 port (2026-05-24): Tweedie overrides this to use
    /// `crate::special::tweedie_series` analytical derivatives directly,
    /// closing the wrong-local-minimum bug from FD on the series.
    fn analytic_shape_score_gradient(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _mu: ndarray::ArrayView1<f64>,
        _dp: f64,
        _n_minus_mp: f64,
        _phi_score: f64,
    ) -> Option<ndarray::Array1<f64>> {
        None
    }

    /// Analytic Hessian of the score w.r.t. THIS loss's shape parameters,
    /// evaluated at the current shape values. Returns `Some(hess)` of shape
    /// `(n_shape, n_shape)` if the family provides closed-form 2nd
    /// derivatives; `None` means the caller falls back to FD on the
    /// analytic gradient (with frozen β̂, the v0.x recipe).
    ///
    /// Inputs match `analytic_shape_score_gradient` plus optional access
    /// to the converged β̂ via the caller — for the cases we currently
    /// port from v0.x (Tweedie via `tweedie_dd_level1` + Wright-series
    /// second derivatives) only the converged μ̂ and the family's own
    /// shape state are needed, so the trait surface stays the same.
    ///
    /// Default `None` keeps families on the FD-on-analytic-gradient path
    /// — that path is already structurally correct (it differentiates the
    /// analytic envelope gradient) and matches v0.x's
    /// `tweedie_theta_grad_hess_analytic` recipe.
    fn analytic_shape_score_hessian(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _mu: ndarray::ArrayView1<f64>,
        _dp: f64,
        _n_minus_mp: f64,
        _phi_score: f64,
    ) -> Option<ndarray::Array2<f64>> {
        None
    }

    /// Per-row observed PIRLS `(W, z)` pair, OR `None` to fall through to
    /// the standard Fisher / Newton-α paths. Used by families whose
    /// deviance isn't an exponential family (scat / TDist) so the
    /// Fisher / Newton-α weights aren't aligned with the observed
    /// `∂²D/∂μ²` — port of mgcv R's `gam.fit4.r:368-369` direct
    /// `w = 0.5·dd$Dmu2·(dμ/dη)²` build.
    ///
    /// Returns parallel vectors `(W, z)` of length `y.len()`. Required
    /// fallback (mgcv R `gam.fit4.r:392-399`): when the observed curvature
    /// is non-positive on a row, substitute the family's **expected**
    /// curvature `½·E[D_μμ]·(dμ/dη)²` and use the Fisher working response
    /// `z = η + (y − μ)·g'(μ)` for that row only. Prior weights MUST be
    /// baked into `W` (PIRLS multiplies in `prior_w` itself for the
    /// Fisher / Newton-α paths; this method is responsible when it owns
    /// the W formula).
    ///
    /// Default returns `None` — every existing gamrs family stays on
    /// the Fisher / Newton-α paths. TDist overrides this to align with
    /// mgcv R's `scat$Dd` and the `gdi2` Level-2 Hessian convention; that
    /// alignment makes the Level-2 analytic Hessian path
    /// (`hess_via_ift_level2`) exact-vs-FD instead of the ~ 30 % mixed-
    /// convention gap that comes from running Fisher PIRLS underneath
    /// observed-W Level-1 / Level-2 chains.
    fn irls_observed_pair(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _mu: ndarray::ArrayView1<f64>,
        _eta: ndarray::ArrayView1<f64>,
        _prior_w: ndarray::ArrayView1<f64>,
    ) -> Option<(ndarray::Array1<f64>, ndarray::Array1<f64>)> {
        None
    }

    /// Whether `PirlsInner` should use the Newton (observed-info) IRLS
    /// weights with per-row Fisher fallback rather than pure Fisher
    /// scoring. Default `false` — pure Fisher is correct for canonical-link
    /// exponential families (Gaussian + identity, Poisson + log, Bernoulli
    /// + logit, Gamma + reciprocal).
    ///
    /// Override on losses whose typical link is **non-canonical** (e.g.
    /// `InverseGaussian` + log, whose canonical link is `1/μ²`). For those
    /// pairs mgcv runs full Newton on the deviance with α = 1 + (y-μ)·(
    /// V'(μ)/V(μ) + g''(μ)/g'(μ)) and falls back to Fisher per-row when
    /// `α ≤ 0` (e.g. ~43% of obs for InverseGaussian + log). PIRLS at
    /// convergence has the same β̂ either way (both are stationary points
    /// of the penalised deviance), but `log|H| = log|X'WX + λS|` uses the
    /// post-convergence W — Fisher vs Newton differs there, and the gap
    /// flows through the REML score and into ρ̂. Reference: v0.x
    /// `src/pirls/mod.rs::Family::is_canonical_link` (lines 446-467) and
    /// `src/pirls/row_step.rs::compute_irls_wz`.
    fn use_newton_irls(&self) -> bool {
        false
    }

    /// Whether this loss is eligible for mgcv_rust's NoRefresh IFT
    /// line-search shortcut (Wood 2011 §4.2; mgcv `gam.fit5.r:367-393`).
    ///
    /// When `true`, the outer-Newton line-search probes may use a
    /// first-order IFT extrapolation `β_trial = β + Σ_k b1[:,k]·Δρ_k`
    /// plus a single working-pair IRLS step at β_trial — skipping inner
    /// PIRLS convergence on every trial λ. The next outer iter's full
    /// eval re-converges β at the accepted λ, so NoRefresh never
    /// corrupts the final fit.
    ///
    /// **Skip list** (matches mgcv_rust `gam_optimized.rs:1512-1518`):
    ///   - **TDist / scat**: W depends on a working `(df, σ²)` state the
    ///     specialised inner fitter (`fit_pirls_tdist`) maintains; one
    ///     IRLS step from IFT-warm β misleads the line-search Armijo.
    ///   - **Quantile**: same as TDist — specialised inner fitter.
    ///   - **InverseGaussian** (any link): variance grows as μ³; W = 1/μ
    ///     for log link swings orders-of-magnitude on small β
    ///     perturbations.
    ///   - **Tweedie**: variance grows as μ^p with p ∈ (1,2); same
    ///     concern as IG at the upper end of p.
    ///
    /// **Eligible**: NegBin, Gaussian, Poisson, Bernoulli/Binomial,
    /// Gamma — all have stable W under small β perturbations.
    fn allows_no_refresh(&self) -> bool {
        false
    }

    /// Per-family outer-Newton tuning (convergence tolerances, iteration
    /// cap, step cap). Default = mgcv-parity (`5e-7 / 1e-7`). Families
    /// with materially different convergence behaviour override.
    ///
    /// See [`crate::outer::OuterTuning`] for fields and rationale. The
    /// fit drivers call this once and convert to [`crate::outer::NewtonOpts`].
    fn outer_tuning(&self) -> crate::outer::OuterTuning {
        crate::outer::OuterTuning::mgcv_default()
    }

    /// Per-element initial μ for PIRLS. Default is mgcv's Bernoulli-style
    /// shrinkage `μ_i = (y_i + ȳ) / 2`. Family overrides:
    /// - **Poisson** (log link) → `max(y_i, 0.1)` to keep μ > 0 before
    ///   `link(μ) = log μ` is taken.
    /// - **Gamma** (log link) → `max(y_i, ε)` similarly.
    /// - **Bernoulli** (logit link) → `(y_i + 0.5) / 2` to keep μ ∈ (0, 1).
    ///
    /// The link is applied OUTSIDE this function (in PIRLS) — `initial_mu`
    /// returns μ on the natural scale; PIRLS then maps to η via `link.link()`.
    /// This keeps `Loss` link-agnostic.
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        let n = y.len();
        let y_bar: f64 = y.iter().sum::<f64>() / (n.max(1) as f64);
        y.iter().map(|&yi| (yi + y_bar) * 0.5).collect()
    }
}

/// Layer 2b — link function `g` with `η = g(μ)`.
pub trait Link {
    fn link(&self, mu: f64) -> f64;
    fn inverse_link(&self, eta: f64) -> f64;
    /// `dμ/dη`. Identity → 1. Logit: `μ(1-μ)`.
    fn d_inverse_link(&self, eta: f64) -> f64;
    /// `dη/dμ = g'(μ)`. Identity → 1. Logit: `1/(μ(1-μ))`. Used in the
    /// PIRLS working response `z = η + (y - μ) · g'(μ)`.
    fn d_link_dmu(&self, mu: f64) -> f64;
    /// `d²η/dμ² = g''(μ)`. Default `0.0` matches Identity. Log: `-1/μ²`.
    /// Logit: `(2μ - 1) / (μ(1-μ))²`. Used by the Newton IRLS path in
    /// `PirlsInner` to compute the curvature correction `α = 1 + (y-μ)·(
    /// V'(μ)/V(μ) + g''(μ)/g'(μ))` for non-canonical links (v0.x
    /// `src/pirls/row_step.rs::compute_irls_wz`, lines 67-82).
    fn d2_link_dmu(&self, _mu: f64) -> f64 {
        0.0
    }
    /// `d³η/dμ³ = g'''(μ)`. Default `0.0`. Log: `2/μ³`. Used by the Tk·KK'
    /// gradient correction in `EnvelopeScore` for non-canonical-link
    /// families, where the score's `log|H|` carries an explicit
    /// β-derivative through the W matrix (mirroring v0.x
    /// `src/reml/mod.rs::reml_gradient_mgcv_exact_ift_inner_at_beta`, the
    /// `a1[i]` computation around line 2089).
    fn d3_link_dmu(&self, _mu: f64) -> f64 {
        0.0
    }
    /// `True` when the link is the canonical link for the loss it's paired
    /// with — lets InnerSolver collapse Newton to Fisher. Default `false`
    /// (caller must opt in for each (Loss, Link) pair).
    fn is_canonical(&self) -> bool {
        false
    }
}

/// Layer 2c — `V(μ)`. Gaussian constant → 1. Binomial → `μ(1-μ)`.
pub trait VarianceFn {
    fn variance(&self, mu: f64) -> f64;
    /// `dV/dμ`. Defaults to `0.0` (Gaussian constant variance). Used by the
    /// Newton IRLS path in `PirlsInner` to compute the curvature correction
    /// `α = 1 + (y-μ)·(V'(μ)/V(μ) + g''(μ)/g'(μ))` for non-canonical-link
    /// families (v0.x `src/pirls/row_step.rs::compute_irls_wz`).
    fn d_variance(&self, _mu: f64) -> f64 {
        0.0
    }
    /// `d²V/dμ²`. Defaults to `0.0`. Used by the Tk·KK' gradient
    /// correction for non-canonical-link families — the `xx` term in
    /// v0.x `src/reml/mod.rs:2101` is `V''/V - (V'/V)² + g'''/g' -
    /// (g''/g')²`. For Inverse Gaussian (V=μ³): `V'' = 6μ`.
    fn d2_variance(&self, _mu: f64) -> f64 {
        0.0
    }
    /// Mirror of `Loss::set_shape_params`. For TDist's `TVariance` the
    /// variance σ² is one of the shape params and needs syncing with the
    /// Loss. Default no-op.
    fn set_shape_params(&mut self, _params: &[f64]) {}
}

/// Layer 3 — given fixed smoothing parameters θ, produce `β̂(θ)` + cached
/// linear-system pieces (Cholesky factor, fitted μ, RSS) for reuse by the
/// `ScoreDerivatives` layer. Concrete impls: `GaussianClosedFormInner`
/// (Phase 0, one Cholesky solve since identity link has no IRLS iteration);
/// later phases add `PirlsInner`, `FastRemlInner`, `ElfPirlsInner`,
/// `JointBetaPhiInner`.
///
/// **Score impls MUST go through this trait, not call concrete inner
/// solves directly.** This is the structural defence against "every new
/// family duplicates score.rs".
pub trait InnerSolver {
    /// What the solver returns. Phase 0 only needs the Gaussian
    /// `InnerFit`, but IRLS-iterative impls will widen this with the
    /// converged μ / η / working weights.
    type Fit;

    /// Solve the penalised inner system at the given log-λ vector.
    /// `rho.len() == 1` in Phase 0 (single smoothing parameter).
    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit>;

    /// Solve with an optional warm-start β. The default implementation
    /// ignores the hint and delegates to `fit`. Inner solvers that can
    /// use the hint (PirlsInner via `eta_init = X·beta_warm`) override
    /// this — the NoRefresh line-search shortcut on EnvelopeScore needs
    /// the warm-start to avoid full PIRLS at trial λ.
    fn fit_warm(&self, rho: &Array1<f64>, _beta_warm: Option<&Array1<f64>>) -> Result<Self::Fit> {
        self.fit(rho)
    }

    /// **Single IRLS step** — port of mgcv R `bam(method="fREML")`'s
    /// one-step inner. Build the working pair `(W, z)` at `η = X·β_warm`,
    /// solve `(X'WX + λS)·β = X'Wz` once, return the new fit. No PIRLS
    /// convergence loop / step-halving guards.
    ///
    /// Used by [`fellner_schall_minimize`]: each FS outer iter consumes
    /// only one IRLS step instead of full PIRLS (~2-3 iters); on
    /// large-n GLM fixtures this is ~3-4× cheaper without sacrificing
    /// FS-update correctness (the warm-start β tracks λ̂(t) closely).
    ///
    /// Default implementation delegates to `fit_warm` — full convergence.
    /// PirlsInner overrides to actually run one step.
    ///
    /// [`fellner_schall_minimize`]: crate::outer::fellner_schall_minimize
    fn fit_single_irls(
        &self,
        rho: &Array1<f64>,
        beta_warm: Option<&Array1<f64>>,
    ) -> Result<Self::Fit> {
        self.fit_warm(rho, beta_warm)
    }

    /// **Lazy Newton-A log|H|** at the converged β. Returns `None` when the
    /// family is canonical-link / doesn't opt into Newton-IRLS (the
    /// canonical Fisher H's `log|A|` off the fit's `a_factor` is then the
    /// right object). For non-canonical Newton-IRLS families (NegBin,
    /// InverseGaussian + log, …) returns `Some(log|X'·W_newton·X + λS|)`.
    ///
    /// **Why on the trait, not on `Fit`**: building Newton-A requires
    /// access to the inner solver's `x_design` / `y` / `family` (link,
    /// variance, prior weights) — the `GaussianInnerFit` doesn't carry
    /// those. Default returns `None` so the closed-form Gaussian / ocat /
    /// quantile inners short-circuit without paying the O(p³) cost. Port
    /// of mgcv_rust `src/reml/mod.rs:460-483` (the Newton-A log|H| block
    /// in the REML score evaluator, NOT inside `fit_pirls_cached`).
    #[allow(unused_variables)]
    fn lazy_newton_log_det_h(&self, fit: &Self::Fit, rho: &Array1<f64>) -> Option<f64> {
        None
    }

    /// **Lazy Tk·KK' / IFT inputs** at the converged β. Returns `None`
    /// when not on the Newton-IRLS path (canonical-link Fisher: term
    /// vanishes by envelope). For non-canonical Newton-IRLS families
    /// returns `Some({a1, lev_uw, eta1_per_term, tr_a_newton_inv_s_per_term,
    /// a_newton_inv, ...})` — used by `EnvelopeScore` (IG path) and
    /// `ShapeAwareEnvelopeScore::analytic_shape_grad_via_ift` (NegBin
    /// shape gradient). Port of mgcv_rust
    /// `src/reml/mod.rs::reml_gradient_mgcv_exact_ift_newton_at_beta`
    /// (`src/reml/mod.rs:2347-2487`). Default returns `None`.
    #[allow(unused_variables)]
    fn lazy_tk_kkt_inputs(
        &self,
        fit: &Self::Fit,
        rho: &Array1<f64>,
    ) -> Option<crate::inner::TkKKTInputs> {
        None
    }

    /// Per-term β-chain of `log|A|` for a score whose `A` carries the FISHER
    /// weight: `tr(A⁻¹ X' diag(∂W/∂η · η₁_k) X)`, η₁_k = X·dβ̂/dρ_k from the
    /// observed penalised Hessian. `None` where the inner solver has no such
    /// weight (or already supplies the Newton form via `lazy_tk_kkt_inputs`).
    #[allow(unused_variables)]
    fn fisher_log_det_beta_chain(&self, fit: &Self::Fit, rho: &Array1<f64>) -> Option<Vec<f64>> {
        None
    }
}

/// Coordinate system the score reports in. Used by downstream consumers
/// (outer optimiser, vcov rebuild) to verify they're not comparing
/// quantities across mismatched bases — the structural defence against the
/// closed-form-vs-FD drift bug class described in plan §1.1 and
/// architecture-assumptions.md §D1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordsKind {
    /// Raw basis (no rotation applied).
    Identity,
    /// Basis is in rotated coords; the `&str` names the rotation (e.g.
    /// "stable" for `Sl.initial.repara`). Phase 0 doesn't use this.
    Reparametrised(&'static str),
}

/// Layer 4 — coupled `(value, grad, hess)` of the outer criterion at θ.
/// All three come from the same internal state — they CANNOT drift apart.
/// This is the keystone abstraction in the v2 architecture (plan §3.4):
/// the closed-form-vs-FD drift bug class is structurally impossible because
/// no caller can ask for grad and hess independently and get inconsistent
/// answers.
///
/// `coords()` reports the basis state — callers verify it before using
/// score outputs that depend on the basis (e.g., reconstructing vcov).
pub trait ScoreDerivatives {
    fn dim(&self) -> usize;

    /// What basis the score operates in. See `CoordsKind`.
    fn coords(&self) -> CoordsKind;

    fn value(&self, theta: &Array1<f64>) -> Result<f64>;

    /// Coupled `(value, grad)`. Cheaper than `value_grad_hess` when the
    /// caller doesn't need the Hessian (e.g., a gradient-only convergence
    /// check at iteration 0).
    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)>;

    /// Coupled `(value, grad, hess)`. This is the surface the outer
    /// optimiser uses; callers MUST NOT FD-probe `value_and_grad` to get a
    /// Hessian (that re-creates the very bug class the trait was designed
    /// to prevent). If a `ScoreDerivatives` impl can't produce an analytic
    /// Hessian, it should FD internally — the FD happens with the impl's
    /// own knowledge of which σ²/coords/etc. to hold fixed.
    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)>;

    /// Per-axis step cap on a single Newton step (mgcv-style — see
    /// `mgcv:smooth.r build_outer_search_vector`). Returns `None` to
    /// signal "use the outer's global `max_step` L∞ cap" (the default).
    /// Shape-aware multi-smooth scores override this to per-axis caps:
    /// ρ axes use 5.0, ocat θ uses 0.5, TweedieTheta=2.0, NegBinTheta=0.5,
    /// TDist log σ²=1.0. Closes the saturated-λ over-leap on multi-smooth
    /// fits — see parity report 2026-05-27.
    fn axis_step_caps(&self) -> Option<Vec<f64>> {
        None
    }
    /// Per-axis lower / upper bounds clamped after each accepted step.
    /// Returns `None` for axes without a bound. Used by mgcv's shape-aware
    /// outer Newton (ocat θ ∈ [-10, 10]; TDist log σ² ∈ [lo, hi]).
    fn axis_bounds(&self) -> Option<Vec<(f64, f64)>> {
        None
    }

    /// Diagnostic counters. Concrete scores own a `FitStats` field and
    /// return `Some(&self.stats)` so the outer solver, line-search probes,
    /// and inner solves can bump counters. Test-only scores (`QuadScore`)
    /// keep the default `None` — counters are advisory, not load-bearing.
    fn stats(&self) -> Option<&crate::stats::FitStats> {
        None
    }
}

/// Result of an outer-loop optimisation.
pub struct OuterFit {
    pub theta: Array1<f64>,
    pub value: f64,
    pub grad_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Layer 5 — outer optimiser. Consumes only `ScoreDerivatives`; knows
/// nothing about Loss/Link/Basis/Family. Concrete impls plug-and-play:
/// `NewtonWithHalving` (Phase 0), `BfgsQuasiNewton`, `Brent1D`,
/// `JointBfgsTrustRegion`.
pub trait OuterSolver {
    fn minimize<S: ScoreDerivatives>(&self, score: &S, theta0: Array1<f64>) -> Result<OuterFit>;
}
