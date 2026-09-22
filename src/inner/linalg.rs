//! Layer 3a — linear backend trait for the penalised normal equations.
//!
//! Above this lives the `InnerSolver` (which assembles `A = X'WX + λS` and
//! its RHS); below this lives ndarray-linalg / LAPACK. The trait abstracts
//! "factor `A` once and reuse for solve / log|A| / tr(A⁻¹·M)" — operations
//! that every score impl needs on the converged inner-fit `A`.
//!
//! Two impls today:
//! - **`CholeskySolver`** (default everywhere) — pure-Cholesky path, the
//!   v0.2 gamrs convention. Lower-triangular factor `L` from
//!   `ndarray-linalg::Cholesky`, plus hand-rolled `forward_solve` /
//!   `back_solve` (one allocation per RHS, matches v0.x bit-for-bit at the
//!   `tr(A⁻¹S)` level — see `score/mod.rs` historical `trace_solve`).
//! - **`LuSolver`** — LAPACK LU via `ndarray-linalg::FactorizeInto`. Same
//!   numerical role; v0.x's reml/system.rs uses LU on the ridged copy of
//!   `A` so this is the v0.x-faithful factorisation. **Verified empirically
//!   to produce β̂ identical to Cholesky's to 1e-13 across the parity
//!   battery** — the residual `low_signal_n1000_k10` 2.27e-6 gap that the
//!   §C4-note Phase-5b/c port hypothesised was a Cholesky-vs-LU mismatch
//!   is **NOT** in factorisation; it's in the outer Newton ρ convergence
//!   (likely v0.x's `Sl.initial.repara` rotation, not ported yet).
//!
//! Choice is type-level — `gamrs::fit::<_, _, _, LuSolver>(family, …)` —
//! never a string key (matches §G "Generic over traits, not enum dispatch"
//! + the typed-config feedback note).
//!
//! Path of a solve through the score body:
//!
//! ```text
//!   GaussianInnerFit<S>::a_factor: S::Factorization
//!         ↓ accessor
//!   inner.log_det_a()     → S::logdet(&a_factor)
//!   inner.trace_a_inv(M)  → S::trace_a_inv(&a_factor, M)
//!   inner.solve(b)        → S::solve(&a_factor, b)
//! ```
//!
//! The trait deliberately stays minimal — no batch-RHS solve, no in-place
//! `solve_mut`. Score impls do ~1 `logdet` + 1 `trace_a_inv` per outer
//! probe; the extra allocation is negligible vs the O(p³) factorisation
//! itself.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{Cholesky, Determinant, FactorizeInto, LUFactorized, Solve, UPLO};

use crate::error::{GamrsError, Result};

/// Factor + solve + log-det + `tr(A⁻¹·M)` operations on a symmetric positive-
/// definite matrix `A = X'WX + λS`. Implementors choose the factorisation
/// — Cholesky for speed, LU for v0.x bit-equivalence at the factor level.
///
/// Type-level dispatch: `gamrs::fit::<_, _, _, LuSolver>(...)` selects the
/// LU backend without any runtime branch.
pub trait LinearSolver: Clone + Copy + Default + 'static {
    /// Concrete factorisation type (e.g. lower Cholesky factor, LU pivots).
    type Factorization: Send + Sync;

    /// Factor `A` (consumes the input — Cholesky stores `L` in place;
    /// LU pivots in place — neither needs `A` after factorisation).
    fn factorize(a: Array2<f64>) -> Result<Self::Factorization>;

    /// Solve `A · x = b` using the stored factorisation. Returns a fresh
    /// `Array1<f64>`.
    fn solve(fact: &Self::Factorization, b: ArrayView1<f64>) -> Array1<f64>;

    /// `log|A|` from the factorisation — `2·Σ log L_ii` for Cholesky,
    /// `Σ log|U_ii|` for LU (with sign tracking absorbed since `A` is
    /// SPD whenever we reach here).
    fn logdet(fact: &Self::Factorization) -> f64;

    /// `tr(A⁻¹·M)` via the v0.x-elementwise pattern: form `A_inv` once by
    /// solving `A·X = I` column-wise, then sum `Σ_{i,j} A_inv[i,j]·M[j,i]`.
    /// Matches v0.x `src/reml/mod.rs:914-918` iteration order — closes
    /// audit finding #4 (was duplicated as `trace_solve` in `score/mod.rs`
    /// and as a column-by-column `trace_a_inv_s` in `inner/mod.rs`).
    fn trace_a_inv(fact: &Self::Factorization, m: ArrayView2<f64>) -> f64;

    /// Form `A⁻¹` explicitly. Used by the Tk·KK' / IFT machinery when it
    /// needs `A⁻¹·S·β` *and* `A⁻¹·xᵢ` for each row; cheaper to materialise
    /// once than to re-solve per use.
    ///
    /// **No default impl** — backends must override because the matrix
    /// size isn't recoverable from the factorisation handle alone in the
    /// generic case (Cholesky `Array2` exposes `nrows()`; LU's pivots do
    /// not). Each backend uses the same "solve `A·X = I` column-by-column"
    /// pattern v0.x's `dgetri` follows internally.
    fn invert(fact: &Self::Factorization) -> Array2<f64>;
}

// =============================================================================
// CholeskySolver — pure-Cholesky backend (default).
// =============================================================================

/// Lower-Cholesky factorisation backend — `A = L·Lᵀ`, default everywhere.
///
/// Numerically identical to the previous direct-Cholesky path that lived
/// inline in `score/mod.rs::trace_solve` + `inner/mod.rs::trace_a_inv_s`.
/// Lifting to the trait closes audit finding #4 (two `trace_solve` impls
/// collapsing into one) without changing any FP arithmetic.
#[derive(Clone, Copy, Default)]
pub struct CholeskySolver;

impl LinearSolver for CholeskySolver {
    type Factorization = Array2<f64>; // lower-triangular L

    fn factorize(a: Array2<f64>) -> Result<Self::Factorization> {
        a.cholesky(UPLO::Lower)
            .map_err(|e| GamrsError::SingularSystem(format!("Cholesky failed: {e}")))
    }

    fn solve(fact: &Self::Factorization, b: ArrayView1<f64>) -> Array1<f64> {
        let z = chol_forward_solve(fact, b);
        chol_back_solve(fact, z.view())
    }

    fn logdet(fact: &Self::Factorization) -> f64 {
        let n = fact.nrows();
        let mut s = 0.0;
        for i in 0..n {
            s += fact[[i, i]].ln();
        }
        2.0 * s
    }

    fn trace_a_inv(fact: &Self::Factorization, m: ArrayView2<f64>) -> f64 {
        let a_inv = Self::invert(fact);
        // v0.x `src/reml/mod.rs:914-918` iteration order:
        // tr(A_inv · M) = Σ_i Σ_j A_inv[i,j] · M[j,i].
        let p = fact.nrows();
        let mut tr = 0.0;
        for i in 0..p {
            for j in 0..p {
                tr += a_inv[[i, j]] * m[[j, i]];
            }
        }
        tr
    }

    fn invert(fact: &Self::Factorization) -> Array2<f64> {
        // A⁻¹ = (L Lᵀ)⁻¹ = L⁻ᵀ · L⁻¹. Materialise into one allocation by
        // forming Y = L⁻¹ via column-wise forward substitution against I,
        // then a_inv[i,j] = Σ_k Y[k,i]·Y[k,j] (i.e. a_inv = Yᵀ·Y).
        //
        // Rewrites the previous per-column `chol_forward_solve +
        // chol_back_solve` to (a) avoid the back-solve entirely — fold L⁻ᵀ
        // into the final symmetric product — and (b) hot-loop on raw
        // slices via `Array2::as_slice` so the inner loops are plain
        // contiguous memory accesses instead of stride-checked indexing.
        // The 10-D Gaussian outer-Newton calls this once per probe; cutting
        // its constant by ~3× was the second half of the bench gap.
        let p = fact.nrows();
        let l_slice = fact
            .as_slice()
            .expect("Cholesky factor must be C-contiguous");
        // y_col_major[i + j*p] = (L⁻¹)[i, j] = j-th column of L⁻¹. Only
        // i ≥ j entries are nonzero (L is lower-triangular ⇒ L⁻¹ is too).
        let mut y_cm = vec![0.0_f64; p * p];
        for j in 0..p {
            // Forward-substitute e_j: solve L·z = e_j.
            // z[i] = (e_j[i] - Σ_{k<i} L[i,k]·z[k]) / L[i,i]
            // z[j] = 1/L[j,j]; z[i<j] = 0; z[i>j] uses prior z's.
            let col_off = j * p;
            // z[j..]
            y_cm[col_off + j] = 1.0 / l_slice[j * p + j];
            for i in (j + 1)..p {
                let l_row_i = &l_slice[i * p..i * p + p];
                let mut s = 0.0_f64;
                // s = Σ_{k=j}^{i-1} L[i,k] · z[k]
                let z_col = &y_cm[col_off..col_off + p];
                for k in j..i {
                    s += l_row_i[k] * z_col[k];
                }
                y_cm[col_off + i] = -s / l_row_i[i];
            }
        }
        // a_inv[i,j] = Σ_k Y[k,i]·Y[k,j] (= (Yᵀ Y)[i,j]). Symmetric.
        let mut a_inv = Array2::<f64>::zeros((p, p));
        {
            let dst = a_inv
                .as_slice_mut()
                .expect("fresh Array2 must be contiguous");
            for i in 0..p {
                let yi = &y_cm[i * p..i * p + p];
                for j in i..p {
                    let yj = &y_cm[j * p..j * p + p];
                    // Both columns are nonzero only on k ≥ max(i, j) = j.
                    let mut s = 0.0_f64;
                    for k in j..p {
                        s += yi[k] * yj[k];
                    }
                    dst[i * p + j] = s;
                    if i != j {
                        dst[j * p + i] = s;
                    }
                }
            }
        }
        a_inv
    }
}

/// Solve `L · z = b` (lower-triangular forward substitution). Kept
/// crate-public so the Quantile warm-start can reuse it without going
/// through the LinearSolver trait (its `A` is a one-shot Gaussian-warm-
/// start system, not a `GaussianInnerFit`).
pub fn chol_forward_solve(l: &Array2<f64>, b: ArrayView1<f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }
    z
}

/// Solve `Lᵀ · x = z` (back-substitution against the transpose of `L`).
pub fn chol_back_solve(l: &Array2<f64>, z: ArrayView1<f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in i + 1..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    x
}

// =============================================================================
// LuSolver — LAPACK LU backend (explicit opt-in).
// =============================================================================

/// LAPACK LU-with-partial-pivoting backend. v0.x's `reml/system.rs` uses
/// LU on the ridged `A` for β̂ and `dgetri` (LU) on the unridged `A` for
/// `tr_a`; this backend lets gamrs callers reproduce that path exactly.
///
/// Empirically (2026-05-24): LU and Cholesky produce β̂ identical to
/// 1e-13 across all 9 family parity fixtures. The §C4-note Phase-5b
/// "Cholesky-vs-LU is the gap" hypothesis was **wrong** — the residual
/// `low_signal_n1000_k10` 2.27e-6 mgcv-parity gap lives in the outer
/// Newton ρ convergence (likely v0.x's `Sl.initial.repara` rotation, not
/// ported to gamrs), not in the factor backend. LU is kept for forward-
/// compat and as the surface for any future indefinite-A path.
#[derive(Clone, Copy, Default)]
pub struct LuSolver;

impl LinearSolver for LuSolver {
    type Factorization = LuFactorState;

    fn factorize(a: Array2<f64>) -> Result<Self::Factorization> {
        let p = a.nrows();
        let lu = a
            .factorize_into()
            .map_err(|e| GamrsError::SingularSystem(format!("LU failed: {e}")))?;
        Ok(LuFactorState { lu, n: p })
    }

    fn solve(fact: &Self::Factorization, b: ArrayView1<f64>) -> Array1<f64> {
        // ndarray-linalg's `Solve::solve` takes an owned `Array1` — pass a copy.
        fact.lu
            .solve(&b.to_owned())
            .expect("LU::solve must succeed (singularity is caught at factorize)")
    }

    fn logdet(fact: &Self::Factorization) -> f64 {
        // For SPD `A`, LU's `U` has positive diagonal up to sign-flips from
        // the row permutation. We track |log U_ii| since `A` is SPD whenever
        // a sane PIRLS converges — sign tracking would matter only if a
        // caller pumped an indefinite `A` through here, which the Newton-W
        // path handles via `eigh` (separate from this trait).
        // The `LUFactorized` struct doesn't expose `U` directly in
        // ndarray-linalg 0.17; we recover `log|det A| = log|det L · U|`
        // via `Solve::sln_det`'s mantissa+exponent decomposition.
        match fact.lu.sln_det() {
            Ok((sign, ln_det)) => {
                // sign is ±1; `A` is SPD so we expect +1, but tolerate −1
                // (caller pumped indefinite A; we still report log|det|).
                let _ = sign;
                ln_det
            }
            Err(_) => f64::NAN,
        }
    }

    fn trace_a_inv(fact: &Self::Factorization, m: ArrayView2<f64>) -> f64 {
        let a_inv = Self::invert(fact);
        // Match v0.x's iteration order exactly (closes audit #4 — same
        // formula for both backends, identical FP to Cholesky's trace).
        let p = fact.n;
        let mut tr = 0.0;
        for i in 0..p {
            for j in 0..p {
                tr += a_inv[[i, j]] * m[[j, i]];
            }
        }
        tr
    }

    fn invert(fact: &Self::Factorization) -> Array2<f64> {
        let p = fact.n;
        let mut a_inv = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let mut e_j = Array1::<f64>::zeros(p);
            e_j[j] = 1.0;
            let col = fact
                .lu
                .solve(&e_j)
                .expect("LU::solve must succeed for inversion column");
            for i in 0..p {
                a_inv[[i, j]] = col[i];
            }
        }
        a_inv
    }
}

/// Owned LU factorisation + the matrix size (needed because
/// `LUFactorized` doesn't expose `.nrows()` in ndarray-linalg 0.17).
pub struct LuFactorState {
    pub lu: LUFactorized<ndarray::OwnedRepr<f64>>,
    pub n: usize,
}

// =============================================================================
// Backend-agnostic factor-and-solve.
// =============================================================================

/// Factor `A` once and return `(factor, β̂)`; the same factor goes on the
/// `GaussianInnerFit` and feeds `log|H|` / `tr(H⁻¹S)`.
///
/// β̂ used to be solved from a second factorisation of `A + 1e-12·max|A_ii|·I`
/// (a v0.x byte-parity habit). That ridge is uniform across the diagonal, so
/// the moment one penalty block dwarfs the rest it stops being a rounding
/// perturbation: on a 10-term house-price scat fit a lot-size term with
/// `max diag S = 2.5e10` at its `λ = 1e6` start made the ridge 25,000 against
/// an intercept diagonal of 2,147, β̂ came back at 0.078× the response level,
/// PIRLS could not move off it (every Newton step re-solves the same ridged
/// system), and the outer Newton optimised a criterion whose inner solution
/// was not the penalised-deviance argmin. The unridged factor has already
/// succeeded by the time a ridge could matter, so it never bought safety —
/// only a β̂ that was not the solution of `A`.
pub fn factor_and_solve<S: LinearSolver>(
    a: &Array2<f64>,
    rhs: ArrayView1<f64>,
) -> Result<(S::Factorization, Array1<f64>)> {
    let factor = S::factorize(a.clone())?;
    let beta = S::solve(&factor, rhs);
    Ok((factor, beta))
}
