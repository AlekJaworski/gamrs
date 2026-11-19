//! Special functions — `log Γ(x)` (Lanczos), `ψ(x)` (digamma), `ψ'(x)` (trigamma).
//!
//! Ported from v0.x `src/pirls/mod.rs:756-963` so gamrs doesn't depend on
//! v0.x and remains WASM-friendly (no native libm needed). Accuracy
//! ~1e-13 across the practical range.

/// `log Γ(x)`. Lanczos approximation with reflection for x < 0.5.
pub fn log_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: Γ(x)·Γ(1-x) = π / sin(πx).
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - log_gamma(1.0 - x);
    }
    let g = 7.0;
    let coef = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut a = coef[0];
    for (i, &c) in coef.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Digamma `ψ(x) = d/dx ln Γ(x)`. Recurrence to push x ≥ 6, then asymptotic.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let xinv = 1.0 / x;
    let xinv2 = xinv * xinv;
    result += x.ln() - 0.5 * xinv;
    let mut t = xinv2;
    result -= t / 12.0;
    t *= xinv2;
    result += t / 120.0;
    t *= xinv2;
    result -= t / 252.0;
    t *= xinv2;
    result += t / 240.0;
    result
}

/// Trigamma `ψ'(x) = d²/dx² ln Γ(x)`. Recurrence to push x ≥ 6, then
/// asymptotic. Ported verbatim from `src/pirls/mod.rs::trigamma` so gamrs
/// matches v0.x byte-for-byte on the Gamma profile-φ Newton iteration.
pub fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let xinv = 1.0 / x;
    let xinv2 = xinv * xinv;
    result += xinv + 0.5 * xinv2;
    let mut t = xinv2 * xinv; // 1/x^3
    result += t / 6.0; // B2 = 1/6
    t *= xinv2;
    result -= t / 30.0; // B4 = -1/30
    t *= xinv2;
    result += t / 42.0; // B6 = 1/42
    t *= xinv2;
    result -= t / 30.0; // B8 = -1/30
    result
}

/// Tweedie series: returns `(log_W, ∂log_W/∂ρ, ∂²log_W/∂ρ², ∂log_W/∂p)`
/// per observation, where `ρ = log φ`. Ported verbatim from v0.x
/// `src/pirls/tweedie.rs::tweedie_series` (v0.2 Phase-1 port,
/// 2026-05-24).
///
/// The j-only pieces of the series (`log_gamma`, `digamma`) are cached
/// across observations — load-bearing for performance (~30 special-fn
/// calls vs ~8000 without cache).
///
/// Only `1 < p < 2` is supported (compound-Poisson-Gamma region).
/// Returns zeros for y ≤ 0 (no series needed at the point mass at 0).
pub fn tweedie_series(y: &[f64], phi: f64, p: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let s = tweedie_series_impl(y, phi, p, false);
    (s.log_w, s.d_rho, s.d2_rho2, s.d_p)
}

/// Per-observation `log W` and its derivatives in `ρ = log φ` and `p`.
pub struct TweedieSeries {
    pub log_w: Vec<f64>,
    pub d_rho: Vec<f64>,
    pub d2_rho2: Vec<f64>,
    pub d_p: Vec<f64>,
    /// Empty unless asked for.
    pub d2_rho_p: Vec<f64>,
    /// Empty unless asked for.
    pub d2_p2: Vec<f64>,
}

/// [`tweedie_series`] plus `∂²log W/∂ρ∂p` and `∂²log W/∂p²`.
pub fn tweedie_series_d2(y: &[f64], phi: f64, p: f64) -> TweedieSeries {
    tweedie_series_impl(y, phi, p, true)
}

fn tweedie_series_impl(y: &[f64], phi: f64, p: f64, second: bool) -> TweedieSeries {
    let n = y.len();
    let mut log_w_out = vec![0.0f64; n];
    let mut dlog_w_drho_out = vec![0.0f64; n];
    let mut d2_log_w_drho2_out = vec![0.0f64; n];
    let mut dlog_w_dp_out = vec![0.0f64; n];
    let n2 = if second { n } else { 0 };
    let mut d2_rho_p_out = vec![0.0f64; n2];
    let mut d2_p2_out = vec![0.0f64; n2];

    let onep = 1.0 - p;
    let onep2 = onep * onep; // (1-p)^2
    let inv_onep2 = 1.0 / onep2; // (1/(1-p))² — d²logW/drho² scale factor
    let twop = 2.0 - p;
    let alpha = twop / onep; // (2-p)/(1-p), negative for 1<p<2
    let rho = phi.ln();

    // w_base = j * (alpha*log(p-1) + rho/onep - log(2-p))
    let log_pm1 = (p - 1.0).ln(); // = log(-onep) = log(p-1)
    let w_base = alpha * log_pm1 + rho / onep - twop.ln();
    let wb1_base = -1.0 / onep; // multiply by j to get wb1_j
    let wp_base = (log_pm1 + rho) / onep2 - alpha / onep + 1.0 / twop;
    let onep3 = onep2 * onep;
    let wpp_base =
        1.0 / ((p - 1.0) * onep2) + 2.0 * (log_pm1 + rho) / onep3 - 1.0 / onep3 - alpha / onep2
            + 1.0 / (twop * twop);

    let log_eps = f64::EPSILON * f64::EPSILON; // ~5e-32
    let log_eps_ln = log_eps.ln();

    // Precompute the j-only pieces. Bound the cache to the maximum j any
    // observation will need (mode = y^(2-p)/(phi·(2-p))) plus a buffer
    // for the upsweep convergence tail.
    let mut j_max_obs: i64 = 1;
    for &yi in y.iter() {
        if yi > 0.0 {
            let xj = yi.powf(twop) / (phi * twop);
            let j_max = (xj.floor() as i64).max(1);
            if j_max > j_max_obs {
                j_max_obs = j_max;
            }
        }
    }
    let j_cache_size = (j_max_obs + 64).max(64) as usize;
    let mut wj0 = vec![0.0_f64; j_cache_size + 1];
    let mut wp1_const = vec![0.0_f64; j_cache_size + 1];
    let mut wpp_const = vec![0.0_f64; if second { j_cache_size + 1 } else { 0 }];
    // ∂²log w_j/∂p², less its `−2j·log y/(1−p)³` row part.
    let wpp_at_j = |jf: f64| -> f64 {
        let a = -jf * alpha;
        jf * wpp_base + jf * (-jf * trigamma(a) / (onep2 * onep2) + 2.0 * digamma(a) / onep3)
    };
    for j in 1..=j_cache_size {
        let jf = j as f64;
        let neg_j_alpha = -jf * alpha;
        wj0[j] = jf * w_base - log_gamma(jf + 1.0) - log_gamma(neg_j_alpha);
        wp1_const[j] = jf * wp_base + (jf / onep2) * digamma(neg_j_alpha);
        if second {
            wpp_const[j] = wpp_at_j(jf);
        }
    }
    let wpp_const_at = |j: usize| -> f64 {
        if j <= j_cache_size {
            wpp_const[j]
        } else {
            wpp_at_j(j as f64)
        }
    };

    // Closures: fall back to direct evaluation past the cache.
    let wj0_at = |j: usize| -> f64 {
        if j <= j_cache_size {
            wj0[j]
        } else {
            let jf = j as f64;
            jf * w_base - log_gamma(jf + 1.0) - log_gamma(-jf * alpha)
        }
    };
    let wp1_const_at = |j: usize| -> f64 {
        if j <= j_cache_size {
            wp1_const[j]
        } else {
            let jf = j as f64;
            jf * wp_base + (jf / onep2) * digamma(-jf * alpha)
        }
    };

    for i in 0..n {
        let yi = y[i];
        if yi <= 0.0 {
            log_w_out[i] = 0.0;
            dlog_w_drho_out[i] = 0.0;
            d2_log_w_drho2_out[i] = 0.0;
            dlog_w_dp_out[i] = 0.0;
            continue;
        }

        let logy_i = yi.ln();
        let alogy_i = alpha * logy_i;
        let logy1p2_i = logy_i / onep2;

        let x = yi.powf(twop) / (phi * twop);
        let j_max = (x.floor() as i64).max(1);
        let wmax_j = j_max as usize;
        let wmax_val = wj0_at(wmax_j) - (j_max as f64) * alogy_i;
        let wmin = wmax_val + log_eps_ln;

        let mut wi = 0.0f64;
        let mut w1i = 0.0f64;
        let mut s2i = 0.0f64;
        let mut wdlogwdp = 0.0f64;
        let (mut s_jp, mut s_p2, mut s_pp) = (0.0f64, 0.0f64, 0.0f64);
        let logy3 = 2.0 * logy_i / onep3;

        // Upsweep from j_max upward.
        let mut j = wmax_j;
        loop {
            let jf = j as f64;
            let wj = wj0_at(j) - jf * alogy_i;
            let wj_scaled = (wj - wmax_val).exp();
            wi += wj_scaled;
            w1i += wj_scaled * jf * wb1_base;
            s2i += wj_scaled * jf * jf;
            let wp1 = wp1_const_at(j) - jf * logy1p2_i;
            wdlogwdp += wj_scaled * wp1;
            if second {
                s_jp += wj_scaled * jf * wp1;
                s_p2 += wj_scaled * wp1 * wp1;
                s_pp += wj_scaled * (wpp_const_at(j) - jf * logy3);
            }
            if wj < wmin {
                break;
            }
            j += 1;
            if j > 10_000_000 {
                break;
            }
        }

        // Downsweep from j_max-1 down to 1.
        if wmax_j >= 1 {
            let mut j_ds: i64 = wmax_j as i64 - 1;
            while j_ds >= 1 {
                let jf = j_ds as f64;
                let ju = j_ds as usize;
                let wj = wj0_at(ju) - jf * alogy_i;
                let wj_scaled = (wj - wmax_val).exp();
                wi += wj_scaled;
                w1i += wj_scaled * jf * wb1_base;
                s2i += wj_scaled * jf * jf;
                let wp1 = wp1_const_at(ju) - jf * logy1p2_i;
                wdlogwdp += wj_scaled * wp1;
                if second {
                    s_jp += wj_scaled * jf * wp1;
                    s_p2 += wj_scaled * wp1 * wp1;
                    s_pp += wj_scaled * (wpp_const_at(ju) - jf * logy3);
                }
                if wj < wmin {
                    break;
                }
                j_ds -= 1;
            }
        }

        log_w_out[i] = wmax_val + wi.ln();
        dlog_w_drho_out[i] = -w1i / wi;
        let e_j = -onep * w1i / wi;
        let e_j_sq = s2i / wi;
        d2_log_w_drho2_out[i] = inv_onep2 * (e_j_sq - e_j * e_j);
        dlog_w_dp_out[i] = wdlogwdp / wi;
        if second {
            // E over the series weights: log W'' = E[ℓ''] + Var/Cov of ℓ'.
            let e_jj = -w1i / wi * onep;
            let e_p = wdlogwdp / wi;
            d2_rho_p_out[i] = e_jj / onep2 + (s_jp / wi - e_jj * e_p) / onep;
            d2_p2_out[i] = s_pp / wi + s_p2 / wi - e_p * e_p;
        }
    }

    TweedieSeries {
        log_w: log_w_out,
        d_rho: dlog_w_drho_out,
        d2_rho2: d2_log_w_drho2_out,
        d_p: dlog_w_dp_out,
        d2_rho_p: d2_rho_p_out,
        d2_p2: d2_p2_out,
    }
}

/// Tweedie saturated log-W per observation — `log W(y; φ, p)` summed over
/// the Dunn-Smyth series. Mirrors v0.x `src/pirls/tweedie.rs::tweedie_series`
/// but without the rho/p derivatives (those come from FD over shape
/// params at the `ShapeAwareEnvelopeScore` layer).
///
/// Only `1 < p < 2` is supported (the practical compound-Poisson-Gamma
/// region). Returns 0 for y ≤ 0 (no series needed).
pub fn tweedie_log_w(y: f64, phi: f64, p: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    if p <= 1.0 || p >= 2.0 {
        return 0.0; // Out-of-domain — caller's responsibility.
    }
    let onep = 1.0 - p; // negative
    let twop = 2.0 - p;
    let alpha = twop / onep; // negative
    let rho = phi.ln();
    let log_pm1 = (p - 1.0).ln();
    let w_base = alpha * log_pm1 + rho / onep - twop.ln();
    let log_eps_ln = (f64::EPSILON * f64::EPSILON).ln();

    let logy = y.ln();
    let alogy = alpha * logy;

    let x = y.powf(twop) / (phi * twop);
    let j_max = (x.floor() as i64).max(1);
    let wmax_j = j_max as usize;
    let wmax_val = {
        let jf = j_max as f64;
        jf * w_base - log_gamma(jf + 1.0) - log_gamma(-jf * alpha) - (j_max as f64) * alogy
    };
    let wmin = wmax_val + log_eps_ln;

    let mut wi = 0.0_f64;
    // Upsweep — cap at 1000 iterations from the mode upward. For
    // well-conditioned (φ, p, y) this is plenty (typically <20 iters);
    // the cap is a safety bound against pathological probes.
    let j_cap = wmax_j + 1000;
    let mut j = wmax_j;
    loop {
        let jf = j as f64;
        let wj = jf * w_base - log_gamma(jf + 1.0) - log_gamma(-jf * alpha) - jf * alogy;
        wi += (wj - wmax_val).exp();
        if wj < wmin || j > j_cap {
            break;
        }
        j += 1;
    }
    // Downsweep
    if wmax_j >= 2 {
        let mut j_ds: i64 = wmax_j as i64 - 1;
        while j_ds >= 1 {
            let jf = j_ds as f64;
            let wj = jf * w_base - log_gamma(jf + 1.0) - log_gamma(-jf * alpha) - jf * alogy;
            wi += (wj - wmax_val).exp();
            if wj < wmin {
                break;
            }
            j_ds -= 1;
        }
    }
    wmax_val + wi.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_gamma_known_values() {
        // log Γ(1) = log(1) = 0
        // log Γ(2) = log(1!) = 0
        // log Γ(3) = log(2!) = log 2
        // log Γ(5) = log(4!) = log 24
        assert!((log_gamma(1.0) - 0.0).abs() < 1e-12);
        assert!((log_gamma(2.0) - 0.0).abs() < 1e-12);
        assert!((log_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-12);
        assert!((log_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn tweedie_series_log_w_matches_scalar() {
        // The 0th return slot of tweedie_series should match the simpler
        // single-obs `tweedie_log_w` to high precision.
        let ys = [0.0_f64, 0.3, 1.2, 5.0, 12.5];
        for &phi in &[0.5_f64, 1.0, 2.0] {
            for &p in &[1.3_f64, 1.5, 1.7] {
                let (lw, _, _, _) = tweedie_series(&ys, phi, p);
                for (i, &yi) in ys.iter().enumerate() {
                    let want = tweedie_log_w(yi, phi, p);
                    assert!(
                        (lw[i] - want).abs() < 1e-9,
                        "y={yi} phi={phi} p={p}: series={} scalar={}",
                        lw[i],
                        want,
                    );
                }
            }
        }
    }

    #[test]
    fn tweedie_series_dlog_w_drho_matches_fd() {
        // Central FD over ρ = log φ.
        let ys = [0.3_f64, 1.2, 5.0, 12.5];
        for &phi in &[0.5_f64, 1.0, 2.0] {
            for &p in &[1.3_f64, 1.5, 1.7] {
                let rho = phi.ln();
                let h = 1e-5;
                let phi_plus = (rho + h).exp();
                let phi_minus = (rho - h).exp();
                let (lw_p, _, _, _) = tweedie_series(&ys, phi_plus, p);
                let (lw_m, _, _, _) = tweedie_series(&ys, phi_minus, p);
                let (_, dr, _, _) = tweedie_series(&ys, phi, p);
                for i in 0..ys.len() {
                    let fd = (lw_p[i] - lw_m[i]) / (2.0 * h);
                    let rel = (dr[i] - fd).abs() / (fd.abs() + 1.0);
                    assert!(
                        rel < 1e-5,
                        "y={} phi={} p={}: analytic={} fd={}",
                        ys[i],
                        phi,
                        p,
                        dr[i],
                        fd,
                    );
                }
            }
        }
    }

    #[test]
    fn tweedie_series_second_derivatives_match_fd() {
        let ys = [0.3_f64, 1.2, 5.0, 12.5];
        for &phi in &[0.5_f64, 1.0, 2.0] {
            for &p in &[1.1_f64, 1.5, 1.9] {
                // Five-point stencil: a narrow step amplifies the series' ~1e-9 noise.
                let h = 1e-3;
                let c = tweedie_series_d2(&ys, phi, p);
                let at = |k: f64| tweedie_series_d2(&ys, phi, p + k * h);
                let (m2, m1, p1, p2) = (at(-2.0), at(-1.0), at(1.0), at(2.0));
                let fd = |f: &dyn Fn(&TweedieSeries) -> f64| {
                    (f(&m2) - f(&p2) + 8.0 * (f(&p1) - f(&m1))) / (12.0 * h)
                };
                for i in 0..ys.len() {
                    let fd_rp = fd(&|s| s.d_rho[i]);
                    let fd_pp = fd(&|s| s.d_p[i]);
                    for (got, fd) in [(c.d2_rho_p[i], fd_rp), (c.d2_p2[i], fd_pp)] {
                        assert!(
                            (got - fd).abs() / (fd.abs() + 1.0) < 1e-6,
                            "y={} phi={phi} p={p}: analytic={got} fd={fd}",
                            ys[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tweedie_series_dlog_w_dp_matches_fd() {
        let ys = [0.3_f64, 1.2, 5.0, 12.5];
        for &phi in &[0.5_f64, 1.0, 2.0] {
            for &p in &[1.3_f64, 1.5, 1.7] {
                let h = 1e-5;
                let (lw_p, _, _, _) = tweedie_series(&ys, phi, p + h);
                let (lw_m, _, _, _) = tweedie_series(&ys, phi, p - h);
                let (_, _, _, dp) = tweedie_series(&ys, phi, p);
                for i in 0..ys.len() {
                    let fd = (lw_p[i] - lw_m[i]) / (2.0 * h);
                    let rel = (dp[i] - fd).abs() / (fd.abs() + 1.0);
                    assert!(
                        rel < 1e-4,
                        "y={} phi={} p={}: analytic={} fd={}",
                        ys[i],
                        phi,
                        p,
                        dp[i],
                        fd,
                    );
                }
            }
        }
    }

    #[test]
    fn digamma_matches_fd_of_log_gamma() {
        let h = 1e-5;
        for &x in &[0.5, 1.0, 2.5, 5.0, 10.0, 100.0] {
            let analytic = digamma(x);
            let fd = (log_gamma(x + h) - log_gamma(x - h)) / (2.0 * h);
            assert!(
                (analytic - fd).abs() < 1e-7,
                "x={x}: analytic={analytic} fd={fd}"
            );
        }
    }
}
