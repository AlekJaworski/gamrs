//! What mgcv's `bam` performance iteration (`bgam.fit`) asks of a family: its
//! `initialize`, link, working weights and pseudo-data, deviance residuals and,
//! for extended families, the θ log-likelihood `estimate.theta` maximises.
//!
//! θ is held in mgcv's own parametrisation (nb: `log θ`; scat:
//! `(log(ν − min.df), log σ)`), since `estimate.theta`'s Newton steps and
//! stopping rule are not invariant to it.

use ndarray::{Array1, Array2, ArrayView1};

use crate::fit::LinkKind;
use crate::special::{digamma, log_gamma, trigamma};
use crate::traits::{Link, Loss, VarianceFn};

use super::{
    Bernoulli, BinomialVariance, ConstantVariance, Family, Gamma, GammaVariance, Gaussian,
    IdentityLink, InverseLink, LogLink, LogitLink, NegBin, NegBinVariance, Poisson,
    PoissonVariance, TDist, TVariance, TDIST_MIN_DF,
};

/// `estimate.theta`'s objective at θ: `D/(2·scale) − ls`, with its θ gradient
/// and Hessian.
pub struct ThetaNll {
    pub value: f64,
    pub grad: Array1<f64>,
    pub hess: Array2<f64>,
}

pub trait BamFamily {
    fn link_kind(&self) -> LinkKind;
    /// mgcv `bam`'s `am`: gaussian identity skips performance iteration.
    fn is_gaussian_identity(&self) -> bool {
        false
    }
    /// `family$initialize`'s `mustart` for one row.
    fn mustart(&self, y: f64, w: f64) -> f64;
    fn linkfun(&self, mu: f64) -> f64;
    fn linkinv(&self, eta: f64) -> f64;
    /// The known scale, or `None` when bam estimates it (Gamma).
    fn known_scale(&self) -> Option<f64> {
        Some(1.0)
    }
    /// Starting θ; empty for a family without one.
    fn initial_theta(&self, _y: ArrayView1<f64>) -> Vec<f64> {
        Vec::new()
    }
    /// `dev.resids` for one row, prior weight included.
    fn dev_resid(&self, y: f64, mu: f64, w: f64, theta: &[f64]) -> f64;
    /// Working response and weight `(z, w)` for one row; `w = 0` drops it.
    fn working(&self, y: f64, eta: f64, mu: f64, w: f64, theta: &[f64]) -> (f64, f64);
    fn theta_nll(
        &self,
        _y: ArrayView1<f64>,
        _mu: ArrayView1<f64>,
        _w: ArrayView1<f64>,
        _theta: &[f64],
    ) -> ThetaNll {
        unreachable!("family has no θ")
    }
    /// `family$aic` as `bam` calls it at the end of the fit; `scale` is the
    /// fitted scale (Gamma's dispersion), ignored elsewhere.
    fn aic(
        &self,
        y: ArrayView1<f64>,
        mu: ArrayView1<f64>,
        w: ArrayView1<f64>,
        theta: &[f64],
        scale: f64,
    ) -> f64;
    /// θ in gamrs's `shape_params` convention.
    fn shape_params(&self, _theta: &[f64]) -> Vec<f64> {
        Vec::new()
    }
}

/// GLM working quantities (`bgam.fit`'s non-extended branch).
fn glm_working<K: Link, V: VarianceFn>(
    link: &K,
    var: &V,
    y: f64,
    eta: f64,
    mu: f64,
    w: f64,
) -> (f64, f64) {
    let mu_eta = link.d_inverse_link(eta);
    if !(w > 0.0 && mu_eta != 0.0) {
        return (0.0, 0.0);
    }
    (
        eta + (y - mu) / mu_eta,
        w * mu_eta * mu_eta / var.variance(mu),
    )
}

impl BamFamily for Family<Gaussian, IdentityLink, ConstantVariance> {
    fn aic(
        &self,
        _y: ArrayView1<f64>,
        _mu: ArrayView1<f64>,
        _w: ArrayView1<f64>,
        _theta: &[f64],
        _scale: f64,
    ) -> f64 {
        unreachable!("gaussian identity reports fast.REML.fit's own REML")
    }
    fn link_kind(&self) -> LinkKind {
        LinkKind::Identity
    }
    fn is_gaussian_identity(&self) -> bool {
        true
    }
    fn mustart(&self, y: f64, _w: f64) -> f64 {
        y
    }
    fn linkfun(&self, mu: f64) -> f64 {
        mu
    }
    fn linkinv(&self, eta: f64) -> f64 {
        eta
    }
    fn known_scale(&self) -> Option<f64> {
        None
    }
    fn dev_resid(&self, y: f64, mu: f64, w: f64, _theta: &[f64]) -> f64 {
        w * (y - mu).powi(2)
    }
    fn working(&self, y: f64, eta: f64, mu: f64, w: f64, _theta: &[f64]) -> (f64, f64) {
        glm_working(&self.link, &self.variance, y, eta, mu, w)
    }
}

impl BamFamily for Family<Poisson, LogLink, PoissonVariance> {
    fn aic(
        &self,
        y: ArrayView1<f64>,
        mu: ArrayView1<f64>,
        w: ArrayView1<f64>,
        _theta: &[f64],
        _scale: f64,
    ) -> f64 {
        -2.0 * (0..y.len())
            .map(|i| w[i] * (y[i] * mu[i].ln() - mu[i] - log_gamma(y[i] + 1.0)))
            .sum::<f64>()
    }
    fn link_kind(&self) -> LinkKind {
        LinkKind::Log
    }
    fn mustart(&self, y: f64, _w: f64) -> f64 {
        y + 0.1
    }
    fn linkfun(&self, mu: f64) -> f64 {
        mu.ln()
    }
    fn linkinv(&self, eta: f64) -> f64 {
        eta.exp().max(f64::EPSILON)
    }
    fn dev_resid(&self, y: f64, mu: f64, w: f64, _theta: &[f64]) -> f64 {
        w * self.loss.deviance_per_obs(y, mu)
    }
    fn working(&self, y: f64, eta: f64, mu: f64, w: f64, _theta: &[f64]) -> (f64, f64) {
        glm_working(&self.link, &self.variance, y, eta, mu, w)
    }
}

impl BamFamily for Family<Bernoulli, LogitLink, BinomialVariance> {
    fn aic(
        &self,
        y: ArrayView1<f64>,
        mu: ArrayView1<f64>,
        w: ArrayView1<f64>,
        _theta: &[f64],
        _scale: f64,
    ) -> f64 {
        // mgcv passes n = 1, so the binomial sizes are the prior weights.
        -2.0 * (0..y.len())
            .filter(|&i| w[i] > 0.0)
            .map(|i| {
                let m = w[i].round();
                let k = (w[i] * y[i]).round();
                log_gamma(m + 1.0) - log_gamma(k + 1.0) - log_gamma(m - k + 1.0)
                    + k * mu[i].ln()
                    + (m - k) * (1.0 - mu[i]).ln()
            })
            .sum::<f64>()
    }
    fn link_kind(&self) -> LinkKind {
        LinkKind::Logit
    }
    fn mustart(&self, y: f64, w: f64) -> f64 {
        (w * y + 0.5) / (w + 1.0)
    }
    fn linkfun(&self, mu: f64) -> f64 {
        (mu / (1.0 - mu)).ln()
    }
    fn linkinv(&self, eta: f64) -> f64 {
        self.link.inverse_link(eta)
    }
    fn dev_resid(&self, y: f64, mu: f64, w: f64, _theta: &[f64]) -> f64 {
        w * self.loss.deviance_per_obs(y, mu)
    }
    fn working(&self, y: f64, eta: f64, mu: f64, w: f64, _theta: &[f64]) -> (f64, f64) {
        glm_working(&self.link, &self.variance, y, eta, mu, w)
    }
}

macro_rules! gamma_bam {
    ($link:ty, $kind:ident) => {
        impl BamFamily for Family<Gamma, $link, GammaVariance> {
            fn link_kind(&self) -> LinkKind {
                LinkKind::$kind
            }
            fn mustart(&self, y: f64, _w: f64) -> f64 {
                y
            }
            fn linkfun(&self, mu: f64) -> f64 {
                self.link.link(mu)
            }
            fn linkinv(&self, eta: f64) -> f64 {
                self.link.inverse_link(eta)
            }
            fn known_scale(&self) -> Option<f64> {
                None
            }
            fn aic(
                &self,
                y: ArrayView1<f64>,
                mu: ArrayView1<f64>,
                w: ArrayView1<f64>,
                _theta: &[f64],
                scale: f64,
            ) -> f64 {
                // dgamma(y, shape = 1/φ, scale = μφ); mgcv's disp = dev/Σw = φ here.
                let a = 1.0 / scale;
                -2.0 * (0..y.len())
                    .map(|i| {
                        let sc = mu[i] * scale;
                        w[i] * ((a - 1.0) * y[i].ln() - y[i] / sc - log_gamma(a) - a * sc.ln())
                    })
                    .sum::<f64>()
                    + 2.0
            }
            fn dev_resid(&self, y: f64, mu: f64, w: f64, _theta: &[f64]) -> f64 {
                w * self.loss.deviance_per_obs(y, mu)
            }
            fn working(&self, y: f64, eta: f64, mu: f64, w: f64, _theta: &[f64]) -> (f64, f64) {
                glm_working(&self.link, &self.variance, y, eta, mu, w)
            }
        }
    };
}
gamma_bam!(LogLink, Log);
gamma_bam!(InverseLink, Inverse);

/// mgcv `nb()`, log link, θ = `log θ`.
impl BamFamily for Family<NegBin, LogLink, NegBinVariance> {
    fn aic(
        &self,
        y: ArrayView1<f64>,
        mu: ArrayView1<f64>,
        w: ArrayView1<f64>,
        theta: &[f64],
        _scale: f64,
    ) -> f64 {
        let th = theta[0].exp();
        2.0 * (0..y.len())
            .map(|i| {
                let (yi, mi) = (y[i], mu[i]);
                w[i] * ((yi + th) * (mi + th).ln() - yi * mi.ln() + log_gamma(yi + 1.0)
                    - th * th.ln()
                    + log_gamma(th)
                    - log_gamma(th + yi))
            })
            .sum::<f64>()
    }
    fn link_kind(&self) -> LinkKind {
        LinkKind::Log
    }
    fn mustart(&self, y: f64, _w: f64) -> f64 {
        if y == 0.0 {
            y + 1.0 / 6.0
        } else {
            y
        }
    }
    fn linkfun(&self, mu: f64) -> f64 {
        mu.ln()
    }
    fn linkinv(&self, eta: f64) -> f64 {
        eta.exp().max(f64::EPSILON)
    }
    fn initial_theta(&self, _y: ArrayView1<f64>) -> Vec<f64> {
        vec![0.0]
    }
    fn dev_resid(&self, y: f64, mu: f64, w: f64, theta: &[f64]) -> f64 {
        let th = theta[0].exp();
        2.0 * w * (y * (y.max(1.0) / mu).ln() - (y + th) * ((y + th) / (mu + th)).ln())
    }
    fn working(&self, y: f64, eta: f64, mu: f64, w: f64, theta: &[f64]) -> (f64, f64) {
        // dDeta with the log link: dμ/dη = μ (floored at eps, as mgcv's mu.eta).
        let th = theta[0].exp();
        let dmu = 2.0 * w * ((y + th) / (mu + th) - y / mu);
        let edmu2 = 2.0 * w * (1.0 / mu - 1.0 / (mu + th));
        let ig1 = mu.max(f64::EPSILON);
        let z = eta - dmu / (edmu2 * ig1);
        let wt = 0.5 * edmu2 * ig1 * ig1;
        if z.is_finite() && wt.is_finite() {
            (z, wt)
        } else {
            (0.0, 0.0)
        }
    }
    fn theta_nll(
        &self,
        y: ArrayView1<f64>,
        mu: ArrayView1<f64>,
        w: ArrayView1<f64>,
        theta: &[f64],
    ) -> ThetaNll {
        let lt = theta[0];
        let th = lt.exp();
        let (mut dev, mut dth, mut dth2) = (0.0, 0.0, 0.0);
        let (mut ls, mut lsth, mut lsth2) = (0.0, 0.0, 0.0);
        for i in 0..y.len() {
            let (yi, mi, wi) = (y[i], mu[i], w[i]);
            dev += self.dev_resid(yi, mi, wi, theta);
            let yth = yi + th;
            let muth = mi + th;
            let r = yth / muth;
            dth += -2.0 * wi * th * (r.ln() + (1.0 - r));
            dth2 += -2.0
                * wi
                * th
                * (r.ln() + th * yth / muth.powi(2) - r - 2.0 * th / muth + 1.0 + th / yth);
            let ylogy = if yi > 0.0 { yi * yi.ln() } else { yi };
            let term = yth * yth.ln() - ylogy + log_gamma(yi + 1.0) - th * th.ln() + log_gamma(th)
                - log_gamma(yth);
            ls -= term * wi;
            let (p0y, p0t) = (digamma(yth), digamma(th));
            lsth -= wi * th * (yth.ln() - p0y + p0t - lt);
            lsth2 -= wi
                * th
                * (yth.ln() - th * trigamma(yth) - p0y + th / yth + th * trigamma(th) + p0t
                    - lt
                    - 1.0);
        }
        ThetaNll {
            value: dev / 2.0 - ls,
            grad: Array1::from_elem(1, dth / 2.0 - lsth),
            hess: Array2::from_elem((1, 1), dth2 / 2.0 - lsth2),
        }
    }
    fn shape_params(&self, theta: &[f64]) -> Vec<f64> {
        vec![theta[0]]
    }
}

/// mgcv `scat()`, identity link, θ = `(log(ν − min.df), log σ)`.
impl BamFamily for Family<TDist, IdentityLink, TVariance> {
    fn aic(
        &self,
        y: ArrayView1<f64>,
        mu: ArrayView1<f64>,
        w: ArrayView1<f64>,
        theta: &[f64],
        _scale: f64,
    ) -> f64 {
        let nu = theta[0].exp() + TDIST_MIN_DF;
        let sig = theta[1].exp();
        let c = -log_gamma((nu + 1.0) / 2.0)
            + log_gamma(nu / 2.0)
            + (sig * (std::f64::consts::PI * nu).sqrt()).ln();
        2.0 * (0..y.len())
            .map(|i| w[i] * (c + (nu + 1.0) * (((y[i] - mu[i]) / sig).powi(2) / nu).ln_1p() / 2.0))
            .sum::<f64>()
    }
    fn link_kind(&self) -> LinkKind {
        LinkKind::Identity
    }
    fn mustart(&self, y: f64, _w: f64) -> f64 {
        if y == 0.0 {
            0.1
        } else {
            y
        }
    }
    fn linkfun(&self, mu: f64) -> f64 {
        mu
    }
    fn linkinv(&self, eta: f64) -> f64 {
        eta
    }
    fn initial_theta(&self, y: ArrayView1<f64>) -> Vec<f64> {
        let n = y.len() as f64;
        let mean = y.sum() / n;
        let sd = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        vec![1.5, (0.8 * sd).ln()]
    }
    fn dev_resid(&self, y: f64, mu: f64, w: f64, theta: &[f64]) -> f64 {
        let nu = theta[0].exp() + TDIST_MIN_DF;
        let sig = theta[1].exp();
        w * (nu + 1.0) * ((((y - mu) / sig).powi(2)) / nu).ln_1p()
    }
    fn working(&self, y: f64, _eta: f64, mu: f64, w: f64, theta: &[f64]) -> (f64, f64) {
        // mgcv's EDmu2 carries no prior weight while Dmu does, so the weight
        // scales the pseudo-data, not the working weight.
        let nu = theta[0].exp() + TDIST_MIN_DF;
        let sig = theta[1].exp();
        let ym = y - mu;
        let a = 1.0 + (ym / sig).powi(2) / nu;
        let dmu = -2.0 * w * (nu + 1.0) * ym / (nu * sig * sig * a);
        let edmu2 = 2.0 * (nu + 1.0) / (sig * sig) / (nu + 3.0);
        (mu - dmu / edmu2, 0.5 * edmu2)
    }
    fn theta_nll(
        &self,
        y: ArrayView1<f64>,
        mu: ArrayView1<f64>,
        w: ArrayView1<f64>,
        theta: &[f64],
    ) -> ThetaNll {
        let nu = theta[0].exp() + TDIST_MIN_DF;
        let sig = theta[1].exp();
        let nu1 = nu + 1.0;
        let nu2 = nu - TDIST_MIN_DF;
        let nu2nu = nu2 / nu;
        let mut dev = 0.0;
        let mut dth = [0.0; 2];
        let mut dth2 = [0.0; 3];
        let mut wsum = 0.0;
        for i in 0..y.len() {
            let wi = w[i];
            let ym = y[i] - mu[i];
            let a = 1.0 + (ym / sig).powi(2) / nu;
            let sig2a = sig * sig * a;
            let nusig2a = nu * sig2a;
            let f = nu1 * ym / nusig2a;
            let f1 = ym / nusig2a;
            let fym = f * ym;
            let f1ym = f1 * ym;
            let fymf1 = fym * f1;
            let ymsig2a = ym / sig2a;
            dev += wi * nu1 * ((ym / sig).powi(2) / nu).ln_1p();
            dth[0] += wi * nu2 * (a.ln() - fym / nu);
            dth[1] += -2.0 * wi * fym;
            dth2[0] += wi
                * (nu2 * a.ln()
                    + nu2nu
                        * ym
                        * ym
                        * (-2.0 * nu2 - nu1 + 2.0 * nu1 * nu2nu - nu1 * nu2nu * f1ym)
                        / nusig2a);
            dth2[1] += 2.0 * wi * (fym - ym * ymsig2a - fymf1 * ym) * nu2nu;
            dth2[2] += 4.0 * wi * fym * (1.0 - f1ym);
            wsum += wi;
        }
        let nu12 = nu1 / 2.0;
        let ls = wsum
            * (log_gamma(nu12)
                - log_gamma(nu / 2.0)
                - (sig * (std::f64::consts::PI * nu).sqrt()).ln());
        let lsth = [
            wsum * (nu2 * digamma(nu12) / 2.0 - nu2 * digamma(nu / 2.0) / 2.0 - 0.5 * nu2nu),
            -wsum,
        ];
        let lsth2_11 = wsum
            * (nu2 * nu2 * trigamma(nu12) / 4.0 + nu2 * digamma(nu12) / 2.0
                - nu2 * nu2 * trigamma(nu / 2.0) / 4.0
                - nu2 * digamma(nu / 2.0) / 2.0
                + 0.5 * nu2nu * nu2nu
                - 0.5 * nu2nu);
        let grad = Array1::from_vec(vec![dth[0] / 2.0 - lsth[0], dth[1] / 2.0 - lsth[1]]);
        let hess = Array2::from_shape_vec(
            (2, 2),
            vec![
                dth2[0] / 2.0 - lsth2_11,
                dth2[1] / 2.0,
                dth2[1] / 2.0,
                dth2[2] / 2.0,
            ],
        )
        .expect("2x2");
        ThetaNll {
            value: dev / 2.0 - ls,
            grad,
            hess,
        }
    }
    fn shape_params(&self, theta: &[f64]) -> Vec<f64> {
        // gamrs: [log σ², log(ν − min.df)].
        vec![2.0 * theta[1], theta[0]]
    }
}
