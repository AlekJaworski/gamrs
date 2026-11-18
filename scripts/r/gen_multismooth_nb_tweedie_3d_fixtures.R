#!/usr/bin/env Rscript
# ============================================================================
# Multi-smooth (2- and 3-margin) NegBin + Tweedie parity fixtures.
#
# REQUIRES R + mgcv (NOT runnable from this gamrs session — the dev machine
# has no R/mgcv installed). To regenerate:
#
#     Rscript scripts/r/gen_multismooth_nb_tweedie_3d_fixtures.R
#
# Companion to scripts/r/gen_multismooth_fixtures.R, which already emits
# the 2-D additive fixtures the layer-3/4 multi-smooth NB/Tweedie parity
# diagnostics consume. This file adds:
#
#   - 3-D additive NegBin (mgcv nb(), profile-theta)
#   - 3-D additive Tweedie profile-p (mgcv tw())
#   - 3-D additive Tweedie fixed-p=1.5 (mgcv Tweedie(p=1.5))
#
# Purpose: the Tk·KK' β-chain term in compute_rho_envelope_gradient only
# kicks in when multiple non-trivial smooths share an η. Ground truth for
# more than two smooths catches off-by-one Tk·KK' bookkeeping that 2-D
# fixtures can mask (e.g. the chain term sums over j ∈ {0,1,2} and a wrong
# index would still cancel out at 2-D by symmetry).
#
# Output format mirrors gen_multismooth_fixtures.R's `emit` helper exactly
# — schema_version=1, inputs.{seed,n,d,k,bs,family,link,method,weights,
# x_train,y_train,...}, mgcv_output.{predictions_train,scale,edf_total,beta}.
# `x_train` is a list of length-d row vectors (jsonlite serialises a Python
# `list[tuple[float, ...]]`-equivalent); the Python loader in
# tests/python/conftest.py reshapes to (n, d) on read.
#
# Family-specific `extra` fields:
#   - nb         → nb_theta_hat
#   - tw         → tweedie_p_hat   (profile-p)
#   - Tweedie(p) → tweedie_p       (fixed-p)
#
# All three fits use cr-margin smooths with k=8 per margin (matches the
# 2-D companion; 3-D ⇒ p_total ≤ 24 which keeps the layer-3 Cholesky fast).
# ============================================================================
suppressMessages({library(mgcv); library(MASS); library(jsonlite)})

FIX <- "tests/fixtures"
k0 <- 8L; k1 <- 8L; k2 <- 8L

emit_3d <- function(path, name, desc, family, link, x0, x1, x2, y, fit,
                    extra = list()) {
  n <- length(y)
  pred <- as.numeric(predict(fit, type = "response"))
  inputs <- list(seed = 20260601L, n = n, d = 3L, k = c(k0, k1, k2),
                 bs = c("cr", "cr", "cr"), family = family, link = link,
                 method = "REML", weights = NULL,
                 x_train = lapply(seq_len(n),
                                  function(i) c(x0[i], x1[i], x2[i])),
                 y_train = as.numeric(y))
  inputs <- c(inputs, extra)
  mo <- list(predictions_train = pred, scale = as.numeric(fit$scale),
             edf_total = as.numeric(sum(fit$edf)),
             beta = as.numeric(coef(fit)))
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv")),
                              generated_at = format(Sys.time(),
                                                    "%Y-%m-%dT%H:%M:%SZ",
                                                    tz = "UTC")),
              inputs = inputs, mgcv_output = mo)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17),
      file = file.path(FIX, path))
  cat(sprintf("  %-46s edf=%.3f scale=%.4f\n", name, sum(fit$edf), fit$scale))
}

# ---- 3-D additive Tweedie (profile-p AND fixed-p share the same y) -------
set.seed(20260601)
n <- 800
x0 <- runif(n); x1 <- runif(n); x2 <- runif(n)
eta <- 0.6 + sin(2 * pi * x0) + 0.5 * (x1 - 0.5) + 0.4 * cos(pi * x2)
mu  <- exp(eta)
y_tw <- rTweedie(mu, p = 1.5, phi = 1.0)
dat_tw <- data.frame(y = y_tw, x0 = x0, x1 = x1, x2 = x2)

cat("Tweedie 3-D:\n")
fit_prof <- gam(y ~ s(x0, k = k0, bs = "cr") + s(x1, k = k1, bs = "cr")
                  + s(x2, k = k2, bs = "cr"),
                family = tw(), method = "REML", data = dat_tw)
p_hat <- as.numeric(fit_prof$family$getTheta(TRUE))
emit_3d("3d_tw_profile_log_n800_k8_cr.json", "3d_tw_profile_log_n800_k8_cr",
        "3-D additive Tweedie, profile-p (mgcv tw()), cr k=8 margins",
        "tw", "log", x0, x1, x2, y_tw, fit_prof,
        list(tweedie_p_hat = p_hat))

fit_fix <- gam(y ~ s(x0, k = k0, bs = "cr") + s(x1, k = k1, bs = "cr")
                 + s(x2, k = k2, bs = "cr"),
               family = Tweedie(p = 1.5, link = "log"),
               method = "REML", data = dat_tw)
emit_3d("3d_tw_fixed_p15_log_n800_k8_cr.json",
        "3d_tw_fixed_p15_log_n800_k8_cr",
        "3-D additive Tweedie, FIXED p=1.5 (mgcv Tweedie(p=1.5)), cr k=8 margins",
        "tw", "log", x0, x1, x2, y_tw, fit_fix, list(tweedie_p = 1.5))

# ---- 3-D additive NegBin (profile-theta) ---------------------------------
cat("NegBin 3-D:\n")
set.seed(20260602)
x0 <- runif(n); x1 <- runif(n); x2 <- runif(n)
eta <- 0.4 + 0.8 * sin(2 * pi * x0) + 0.6 * x1 + 0.5 * (x2 - 0.5)^2
mu  <- exp(eta)
theta_true <- 4.0
y_nb <- rnegbin(n, mu = mu, theta = theta_true)
dat_nb <- data.frame(y = y_nb, x0 = x0, x1 = x1, x2 = x2)
fit_nb <- gam(y ~ s(x0, k = k0, bs = "cr") + s(x1, k = k1, bs = "cr")
                + s(x2, k = k2, bs = "cr"),
              family = nb(), method = "REML", data = dat_nb)
theta_hat <- as.numeric(fit_nb$family$getTheta(TRUE))
emit_3d("3d_nb_log_n800_k8_cr.json", "3d_nb_log_n800_k8_cr",
        "3-D additive NegBin, profile-theta (mgcv nb()), cr k=8 margins",
        "nb", "log", x0, x1, x2, y_nb, fit_nb,
        list(nb_theta_hat = theta_hat))
cat(sprintf("  (nb theta_hat=%.4f)\n", theta_hat))
