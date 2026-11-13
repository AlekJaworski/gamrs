//! Analytic IFT Hessian — direct port of mgcv_rust's
//! `reml_hessian_mgcv_exact_ift` (`nn_exploring/src/reml/mod.rs:2511-2813`).
//!
//! Produces the m×m ρ-Hessian of REML/LAML with **zero PIRLS solves**,
//! using the implicit-function-theorem identities
//!
//! ```text
//!   b1[:,k]      =   ∂β/∂ρ_k   = -λ_k · A⁻¹ S_k β
//!   A · b2[k,j]  =   ∂²β/∂ρ_kρ_j
//!                  = -(λ_j S_j b1[:,k] + λ_k S_k b1[:,j] + δ_kj·λ_k S_k β)
//! ```
//!
//! (mgcv `gdi.c::ift1` deriv2 branch; mgcv_rust ports it inline at
//! `reml/mod.rs:2657-2674`.) The Hessian assembly is then the same
//! `Dp/(2σ²) + det2/2` decomposition mgcv_rust uses (lines 2676-2778),
//! split into seven additive pieces in the `term_*` locals of
//! [`hess_ift_rho`] below.
//!
//! Replaces the prior bespoke gamrs Hessian path which paid `m·n·p²` for
//! the W-chain (`xmj = X·A⁻¹S_jA⁻¹` at envelope.rs:534) and `m·p³` for
//! per-term `A⁻¹S_j` materialisation. This port is `O(m²·p²)`, dominated
//! by m² A⁻¹·b2_kj solves; the trace cross uses dense `A⁻¹S_j` once per
//! term then a single `Σ ainv_s[k]⊙ainv_s[j]ᵀ` reduction.

use ndarray::{Array1, Array2};

/// Inputs to the analytic IFT Hessian helper. Built by the score-side
/// caller from a converged inner fit + the cached score-time quantities.
pub(crate) struct HessIftCtx<'a> {
    /// Per-term penalty matrices `S_j`, each (p, p). Length m.
    pub s_list: &'a [Array2<f64>],
    /// `λ_j = exp(ρ_j)` per term, length m.
    pub lambda: &'a [f64],
    /// Converged β, length p.
    pub beta: &'a Array1<f64>,
    /// `A⁻¹` where `A = X'WX + Σ λ_j S_j`, shape (p, p). Reused from
    /// `GaussianInnerFit::a_inv()`.
    pub a_inv: &'a Array2<f64>,
    /// `X'WX`, shape (p, p). Used for the data-fit `2·X'WX·b1[:,k]` piece
    /// and for the `b1' (A − X'WX) b1` identity (mgcv_rust:2721-2737).
    pub xtwx: &'a Array2<f64>,
    /// σ² score-side dispersion. Divides the data-fit half of the Hessian
    /// (mgcv_rust:2773 `d2_kj / (2 * scale_est)`).
    pub sigma2: f64,
    /// `∂D/∂β` at converged β, length p. For Gaussian / canonical-link
    /// GLMs at convergence this is `-2·λSβ` (working-RSS form) — used
    /// via `dev_grad_beta · b2[k,j]` in the term-2 piece.
    pub dev_grad_beta: &'a Array1<f64>,
}

/// Port of mgcv_rust `reml_hessian_mgcv_exact_ift` (`reml/mod.rs:2511`).
/// Returns the m×m ρ-Hessian of REML/LAML at converged β.
///
/// Math (cited from mgcv_rust line ranges):
///
/// 1. `compute_b1_ift` (mgcv_rust:1772-1793): `b1[:,k] = -λ_k · A⁻¹ S_k β`.
/// 2. `s_beta_per_j` (mgcv_rust:2596-2598): `S_j β` per term.
/// 3. `sum_lambda_s_beta` (mgcv_rust:2600-2606): `Σ_j λ_j S_j β`.
/// 4. `tr_a_s_per_j` + sparse `ainvs_per_j` (mgcv_rust:2608-2638): per-term
///    `tr(A⁻¹ S_k)` plus the columns of `A⁻¹ S_k`. gamrs uses dense `A⁻¹·S_j`
///    (full p×p) because `s_list` stores full p×p matrices.
/// 5. `d2dev_b1` (mgcv_rust:2641-2651): `(∂²D/∂β² · b1)[:,k] = 2 X'WX b1[:,k]`.
/// 6. `assemble_b2` (mgcv_rust:2657-2674): one A⁻¹ multiply per (k,j) pair.
/// 7. Hessian assembly loop (mgcv_rust:2676-2779): seven additive `term_*`
///    pieces summed into `d2_kj`, plus `det2_kj` for the `log|H|` curvature,
///    combined as `h_kj = d2_kj / (2 σ²) + det2_kj / 2`.
///
/// Returns the unscaled REML Hessian — the `+0.5 log|H|` and
/// `-0.5 log|λS|` factors are already inside `det2/2`; the data-fit half
/// already divides by 2σ². Caller adds the Tk·KK' contribution separately
/// if needed (mgcv_rust:2788-2810 — gated on InverseGaussian / Binomial /
/// QuasiBinomial; default-off for NegBin / Gaussian / Poisson).
pub(crate) fn hess_ift_rho(ctx: &HessIftCtx<'_>) -> Array2<f64> {
    let p = ctx.a_inv.nrows();
    let m = ctx.lambda.len();
    debug_assert_eq!(ctx.s_list.len(), m);
    debug_assert_eq!(ctx.beta.len(), p);
    debug_assert_eq!(ctx.xtwx.nrows(), p);
    debug_assert_eq!(ctx.dev_grad_beta.len(), p);

    let beta = ctx.beta;
    let a_inv = ctx.a_inv;
    let xtwx = ctx.xtwx;
    let lambdas = ctx.lambda;
    let s_list = ctx.s_list;
    let sigma2 = ctx.sigma2;
    let dev_grad_beta = ctx.dev_grad_beta;

    // --- IFT first derivatives: b1[:,k] = -λ_k · A⁻¹ S_k β (mgcv_rust:2593, 1772) ---
    let mut b1 = Array2::<f64>::zeros((p, m));
    let mut s_beta_per_j: Vec<Array1<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        let s_k_beta: Array1<f64> = s_list[k].dot(beta);
        let ainv_sk_beta = a_inv.dot(&s_k_beta);
        let lam_k = lambdas[k];
        for r in 0..p {
            b1[[r, k]] = -lam_k * ainv_sk_beta[r];
        }
        s_beta_per_j.push(s_k_beta);
    }

    // ΣλSβ (mgcv_rust:2600-2606).
    let mut sum_lambda_s_beta = Array1::<f64>::zeros(p);
    for j in 0..m {
        let lam_j = lambdas[j];
        for r in 0..p {
            sum_lambda_s_beta[r] += lam_j * s_beta_per_j[j][r];
        }
    }

    // --- Per-term tr(A⁻¹ S_k) and A⁻¹ S_k (full p×p) for cross terms ---
    // mgcv_rust uses sparse (p, k_block) ainvs_per_j because penalties carry
    // explicit block offsets; gamrs's `s_list` stores full p×p matrices,
    // so we form dense A⁻¹·S_j once per term. Cost: m × O(p³); still cheap
    // relative to PIRLS solves and consistent with mgcv_rust's logic.
    let mut tr_a_s_per_j: Vec<f64> = Vec::with_capacity(m);
    let mut ainv_s_per_j: Vec<Array2<f64>> = Vec::with_capacity(m);
    for j in 0..m {
        let ainv_sj: Array2<f64> = a_inv.dot(&s_list[j]);
        // tr(A⁻¹ S_j) = trace of (p×p) product.
        let mut tr_as = 0.0_f64;
        for r in 0..p {
            tr_as += ainv_sj[[r, r]];
        }
        tr_a_s_per_j.push(tr_as);
        ainv_s_per_j.push(ainv_sj);
    }

    // --- d2dev_b1[:,k] = 2·X'WX·b1[:,k] (Fisher d²D/dβ²; mgcv_rust:2644-2651) ---
    let mut d2dev_b1 = Array2::<f64>::zeros((p, m));
    for k in 0..m {
        let b1_k = b1.column(k).to_owned();
        let xtwx_b1_k = xtwx.dot(&b1_k);
        for r in 0..p {
            d2dev_b1[[r, k]] = 2.0 * xtwx_b1_k[r];
        }
    }

    // assemble_b2 closure (mgcv_rust:2657-2674):
    //   b2[k,j] = -A⁻¹·(λ_j S_j b1[:,k] + λ_k S_k b1[:,j] + δ_kj·λ_k S_k β)
    let assemble_b2 = |k: usize, j: usize| -> Array1<f64> {
        let lam_j = lambdas[j];
        let lam_k = lambdas[k];
        let b1_k = b1.column(k).to_owned();
        let b1_j = b1.column(j).to_owned();
        let s_j_b1k = s_list[j].dot(&b1_k);
        let s_k_b1j = s_list[k].dot(&b1_j);
        let mut rhs = Array1::<f64>::zeros(p);
        for r in 0..p {
            rhs[r] = -(lam_j * s_j_b1k[r] + lam_k * s_k_b1j[r]);
        }
        if k == j {
            for r in 0..p {
                rhs[r] -= lam_k * s_beta_per_j[k][r];
            }
        }
        a_inv.dot(&rhs)
    };

    // --- Main Hessian assembly loop (mgcv_rust:2676-2779) ---
    let mut hess = Array2::<f64>::zeros((m, m));
    for k_out in 0..m {
        for j_out in k_out..m {
            let lam_k = lambdas[k_out];
            let lam_j = lambdas[j_out];

            // term_d2dev = b1[:,j]' · d2dev_b1[:,k] (mgcv_rust:2685-2688).
            let mut term_d2dev = 0.0_f64;
            for r in 0..p {
                term_d2dev += b1[[r, j_out]] * d2dev_b1[[r, k_out]];
            }

            let b2_kj = assemble_b2(k_out, j_out);

            // term_dev_b2 = (∂D/∂β)' · b2[k,j] (mgcv_rust:2694-2697).
            let mut term_dev_b2 = 0.0_f64;
            for r in 0..p {
                term_dev_b2 += dev_grad_beta[r] * b2_kj[r];
            }

            // term_kron_bsb = δ_kj · λ_k · β'S_kβ (mgcv_rust:2700-2704).
            let term_kron_bsb = if k_out == j_out {
                let sb = &s_beta_per_j[k_out];
                let bsb_k: f64 = beta.iter().zip(sb.iter()).map(|(a, b)| a * b).sum();
                lam_k * bsb_k
            } else {
                0.0
            };

            // term_lk_skb_b1j = 2 λ_k (S_kβ)' b1[:,j] (mgcv_rust:2707-2711).
            let mut term_lk_skb_b1j = 0.0_f64;
            for r in 0..p {
                term_lk_skb_b1j += s_beta_per_j[k_out][r] * b1[[r, j_out]];
            }
            term_lk_skb_b1j *= 2.0 * lam_k;

            // term_lj_sjb_b1k = 2 λ_j (S_jβ)' b1[:,k] (mgcv_rust:2714-2718).
            let mut term_lj_sjb_b1k = 0.0_f64;
            for r in 0..p {
                term_lj_sjb_b1k += s_beta_per_j[j_out][r] * b1[[r, k_out]];
            }
            term_lj_sjb_b1k *= 2.0 * lam_j;

            // term_b1_sls_b1 = 2·b1[:,j]'·(ΣλS)·b1[:,k]
            //                = 2·(b1[:,j]' A b1[:,k] − b1[:,j]' X'WX b1[:,k])
            // with A·b1[:,k] = −λ_k·S_kβ ⇒ b1[:,j]' A b1[:,k] = −λ_k·b1[:,j]'·(S_kβ).
            // (mgcv_rust:2720-2737.)
            let mut b1j_a_b1k = 0.0_f64;
            for r in 0..p {
                b1j_a_b1k += b1[[r, j_out]] * (-lam_k * s_beta_per_j[k_out][r]);
            }
            let b1_k_col = b1.column(k_out).to_owned();
            let xtwx_b1k = xtwx.dot(&b1_k_col);
            let mut b1j_xtwx_b1k = 0.0_f64;
            for r in 0..p {
                b1j_xtwx_b1k += b1[[r, j_out]] * xtwx_b1k[r];
            }
            let term_b1_sls_b1 = 2.0 * (b1j_a_b1k - b1j_xtwx_b1k);

            // term_sls_b2 = 2·(ΣλSβ)' b2[k,j] (mgcv_rust:2740-2744).
            let mut term_sls_b2 = 0.0_f64;
            for r in 0..p {
                term_sls_b2 += sum_lambda_s_beta[r] * b2_kj[r];
            }
            term_sls_b2 *= 2.0;

            let d2_kj = term_d2dev
                + term_dev_b2
                + term_kron_bsb
                + term_lk_skb_b1j
                + term_lj_sjb_b1k
                + term_b1_sls_b1
                + term_sls_b2;

            // det2_kj (mgcv_rust:2754-2771):
            //   = δ_kj·λ_k·tr(A⁻¹S_k) − λ_k·λ_j·tr(A⁻¹S_k·A⁻¹S_j)
            // tr(A⁻¹S_k·A⁻¹S_j) via the dense `ainv_s_per_j` product:
            //   = Σ_{r,c} (A⁻¹S_k)[r,c] · (A⁻¹S_j)[c,r]
            let ai_k = &ainv_s_per_j[k_out];
            let ai_j = &ainv_s_per_j[j_out];
            let mut tr_cross = 0.0_f64;
            for r in 0..p {
                for c in 0..p {
                    tr_cross += ai_k[[r, c]] * ai_j[[c, r]];
                }
            }
            let det2_kj = if k_out == j_out {
                lam_k * tr_a_s_per_j[k_out] - lam_k * lam_j * tr_cross
            } else {
                -lam_k * lam_j * tr_cross
            };

            let h_kj = d2_kj / (2.0 * sigma2) + det2_kj / 2.0;
            hess[[k_out, j_out]] = h_kj;
            if k_out != j_out {
                hess[[j_out, k_out]] = h_kj;
            }
        }
    }

    hess
}

/// Build `X'WX` from cached `(X, working_weights)`. Used by the envelope
/// + shape-aware paths to feed [`HessIftCtx::xtwx`] when no pre-cached
/// `xtwx` is available on the converged inner fit.
pub(crate) fn build_xtwx(x_design: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    crate::inner::split::SplitDesign::new(x_design.view()).xtwx(w.view())
}

/// Compute `∂D/∂β` at converged β for the IFT Hessian's `term_dev_b2`.
///
/// Port of mgcv_rust `compute_dev_grad_beta` (`reml/mod.rs:1806-1844`):
///
/// - **Gaussian / Gaussian-equivalent** (no `y_original`): working-RSS
///   form `-2·X'·W·(y_input - X·β)`. At converged β with the score's
///   working `(y_input, w)`, this is exactly `-2·X'·W·(z - X·β) = -2·λSβ`
///   by the PIRLS normal equation. (Used by closed-form Gaussian; the
///   resulting `term_dev_b2` vanishes only at the canonical-link optimum.)
/// - **GLM with `y_original`**: `v1[i] = -2·prior_w[i]·(y_orig_i - μ_i) /
///   (V(μ_i)·g'(μ_i))`, then `dev_grad = X' · v1` (mgcv_rust:1830-1843).
///   At converged β this is `-2·λSβ` for the canonical link, and a
///   non-canonical residual otherwise.
///
/// Returns the p-vector.
pub(crate) fn compute_dev_grad_beta_working_rss(
    x_design: &Array2<f64>,
    working_weights: &Array1<f64>,
    working_response: &Array1<f64>,
    beta: &Array1<f64>,
) -> Array1<f64> {
    let n = x_design.nrows();
    let fitted = x_design.dot(beta);
    let mut v1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        v1[i] = -2.0 * working_weights[i] * (working_response[i] - fitted[i]);
    }
    x_design.t().dot(&v1)
}
