//! Shared link functions: Identity, Log, Logit, Inverse.

use crate::traits::Link;

#[derive(Clone)]
pub struct IdentityLink;

#[derive(Clone)]
pub struct LogLink;

#[derive(Clone)]
pub struct LogitLink;

/// Reciprocal link `g(μ) = 1/μ`. Canonical link for the Gamma family
/// (mgcv's `Gamma()` default). Pair with `Gamma` Loss + `GammaVariance`
/// via the `gamma_inverse()` constructor.
#[derive(Clone)]
pub struct InverseLink;

impl Link for IdentityLink {
    fn link(&self, mu: f64) -> f64 {
        mu
    }
    fn inverse_link(&self, eta: f64) -> f64 {
        eta
    }
    fn d_inverse_link(&self, _eta: f64) -> f64 {
        1.0
    }
    fn d_link_dmu(&self, _mu: f64) -> f64 {
        1.0
    }
    fn is_canonical(&self) -> bool {
        // Identity IS the canonical link for Gaussian.
        true
    }
}

impl Link for LogLink {
    fn link(&self, mu: f64) -> f64 {
        let eps = 1e-300;
        mu.max(eps).ln()
    }
    fn inverse_link(&self, eta: f64) -> f64 {
        eta.exp()
    }
    fn d_inverse_link(&self, eta: f64) -> f64 {
        // dμ/dη = exp(η) = μ
        eta.exp()
    }
    fn d_link_dmu(&self, mu: f64) -> f64 {
        // dη/dμ = 1/μ
        let eps = 1e-300;
        1.0 / mu.max(eps)
    }
    fn d2_link_dmu(&self, mu: f64) -> f64 {
        // d²η/dμ² = -1/μ². Floor the DENOMINATOR (not just μ): flooring μ at
        // 1e-300 and then squaring underflows to 0, so `-1/(m*m)` would be
        // -1/0 = -inf. μ=exp(η) can legitimately reach ~0 on a divergent
        // log-link iterate, so cap the denominator to keep d2 large-but-finite.
        let eps = 1e-300;
        let m = mu.max(eps);
        -1.0 / (m * m).max(eps)
    }
    fn d3_link_dmu(&self, mu: f64) -> f64 {
        // d³η/dμ³ = 2/μ³ — same denominator-floor reasoning as d2.
        let eps = 1e-300;
        let m = mu.max(eps);
        2.0 / (m * m * m).max(eps)
    }
    fn d4_link_dmu(&self, mu: f64) -> f64 {
        let eps = 1e-300;
        let m = mu.max(eps);
        -6.0 / (m * m * m * m).max(eps)
    }
    fn is_canonical(&self) -> bool {
        // log IS the canonical link for Poisson.
        true
    }
}

impl Link for LogitLink {
    fn link(&self, mu: f64) -> f64 {
        // logit(μ) = log(μ / (1 - μ)). Clamp at the endpoints.
        let eps = 1e-15;
        let mu = mu.clamp(eps, 1.0 - eps);
        (mu / (1.0 - mu)).ln()
    }
    fn inverse_link(&self, eta: f64) -> f64 {
        // μ = 1/(1+exp(-η)). Numerically stable for large |η|.
        if eta >= 0.0 {
            1.0 / (1.0 + (-eta).exp())
        } else {
            let e = eta.exp();
            e / (1.0 + e)
        }
    }
    fn d_inverse_link(&self, eta: f64) -> f64 {
        // dμ/dη = μ(1-μ).
        let mu = self.inverse_link(eta);
        mu * (1.0 - mu)
    }
    fn d_link_dmu(&self, mu: f64) -> f64 {
        // dη/dμ = 1 / (μ(1-μ)).
        let eps = 1e-15;
        let mu = mu.clamp(eps, 1.0 - eps);
        1.0 / (mu * (1.0 - mu))
    }
    fn d2_link_dmu(&self, mu: f64) -> f64 {
        // d²η/dμ² = (2μ - 1) / (μ(1-μ))²
        let eps = 1e-15;
        let m = mu.clamp(eps, 1.0 - eps);
        let denom = m * (1.0 - m);
        (2.0 * m - 1.0) / (denom * denom)
    }
    fn is_canonical(&self) -> bool {
        // Logit IS the canonical link for Bernoulli/Binomial.
        true
    }
}

impl Link for InverseLink {
    fn link(&self, mu: f64) -> f64 {
        // g(μ) = 1/μ. Floor μ away from 0 to keep η finite.
        let eps = 1e-300;
        1.0 / mu.max(eps)
    }
    fn inverse_link(&self, eta: f64) -> f64 {
        // g⁻¹(η) = 1/η. Floor |η| away from 0 to keep μ finite.
        let eps = 1e-300;
        let sign = if eta < 0.0 { -1.0 } else { 1.0 };
        1.0 / (sign * eta.abs().max(eps))
    }
    fn d_inverse_link(&self, eta: f64) -> f64 {
        // dμ/dη = -1/η²
        let eps = 1e-300;
        let e2 = (eta * eta).max(eps);
        -1.0 / e2
    }
    fn d_link_dmu(&self, mu: f64) -> f64 {
        // dη/dμ = -1/μ²
        let eps = 1e-300;
        let m2 = (mu * mu).max(eps);
        -1.0 / m2
    }
    fn d2_link_dmu(&self, mu: f64) -> f64 {
        // d²η/dμ² = 2/μ³. Floor the DENOMINATOR (matching d_link_dmu): μ floored
        // at 1e-300 then cubed underflows to 0, so `2/(m*m*m)` would be +inf.
        let eps = 1e-300;
        let m = mu.max(eps);
        2.0 / (m * m * m).max(eps)
    }
    fn d3_link_dmu(&self, mu: f64) -> f64 {
        // d³η/dμ³ = -6/μ⁴ — same denominator-floor reasoning as d2.
        let eps = 1e-300;
        let m = mu.max(eps);
        -6.0 / (m * m * m * m).max(eps)
    }
    fn d4_link_dmu(&self, mu: f64) -> f64 {
        let eps = 1e-300;
        let m = mu.max(eps);
        24.0 / (m * m * m * m * m).max(eps)
    }
    fn is_canonical(&self) -> bool {
        // Reciprocal IS the canonical link for Gamma.
        true
    }
}
