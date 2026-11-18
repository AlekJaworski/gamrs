#!/usr/bin/env Rscript
# Multi-smooth (2-D additive) quantile OOS-pinball reference via qgam.
#
#     Rscript scripts/r/gen_quantile_multismooth_fixture.R
#
# gamrs's quantile (ELF) family ports qgam (Fasiolo et al. 2021, built on
# mgcv). The single-smooth quantile already had an OOS-pinball parity test
# vs mgcv_rust; this closes the README's "Quantile/ELF is single-smooth-only"
# gap by giving the *multi-smooth* additive quantile a qgam ground-truth
# reference. OOS pinball is the quality metric that matters for quantiles,
# so we fit qgam on a fixed train split and record its held-out pinball +
# test predictions per τ. The gamrs parity test
# (tests/python/test_parity_multismooth.py) fits gamrs on the identical
# split and asserts its OOS pinball is on par with qgam's.
#
# DGP: 2-D additive heteroskedastic — y = sin(2πx0) + 2(x1−½) + (0.15+0.25 x0)·N(0,1).
# Additive in (x0, x1) with x0-dependent spread, so an additive s(x0)+s(x1)
# quantile is well-specified and the τ-surfaces genuinely fan out.
suppressMessages({library(qgam); library(mgcv); library(jsonlite)})

FIX  <- "tests/fixtures"
SEED <- 20260610L
N_TR <- 600L
N_TE <- 400L
TAUS <- c(0.1, 0.5, 0.9)
K    <- 8L

pinball <- function(y, q, tau) {
  r <- y - q
  mean(pmax(tau * r, (tau - 1) * r))
}

gen <- function(n) {
  x0 <- runif(n)
  x1 <- runif(n)
  f  <- sin(2 * pi * x0) + 2 * (x1 - 0.5)
  sc <- 0.15 + 0.25 * x0
  list(x0 = x0, x1 = x1, y = f + sc * rnorm(n))
}

set.seed(SEED)
tr <- gen(N_TR)
te <- gen(N_TE)
dat_tr <- data.frame(x0 = tr$x0, x1 = tr$x1, y = tr$y)
dat_te <- data.frame(x0 = te$x0, x1 = te$x1, y = te$y)

form <- y ~ s(x0, k = 8, bs = "cr") + s(x1, k = 8, bs = "cr")

cat("qgam 2-D additive quantile:\n")
per_tau <- list()
for (tau in TAUS) {
  fit  <- suppressMessages(qgam(form, data = dat_tr, qu = tau))
  q_te <- as.numeric(predict(fit, newdata = dat_te))
  pb   <- pinball(dat_te$y, q_te, tau)
  per_tau[[as.character(tau)]] <- list(oos_pinball = pb, pred_test = q_te)
  cat(sprintf("  tau=%.2f: qgam OOS pinball=%.6f\n", tau, pb))
}

inputs <- list(
  seed = SEED, n_train = N_TR, n_test = N_TE,
  k = c(K, K), bs = c("cr", "cr"), taus = TAUS,
  x_train = lapply(seq_len(N_TR), function(i) c(tr$x0[i], tr$x1[i])),
  y_train = tr$y,
  x_test  = lapply(seq_len(N_TE), function(i) c(te$x0[i], te$x1[i])),
  y_test  = te$y
)
obj <- list(
  schema_version = 1L,
  name = "2d_quantile_oos_hetero_n600_k8_cr",
  description = "2-D additive heteroskedastic quantile OOS-pinball reference (qgam)",
  metadata = list(
    engine = "qgam",
    qgam_version = as.character(packageVersion("qgam")),
    mgcv_version = as.character(packageVersion("mgcv")),
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  ),
  inputs = inputs,
  qgam_output = list(per_tau = per_tau)
)
cat(toJSON(obj, auto_unbox = TRUE, digits = 17),
    file = file.path(FIX, "2d_quantile_oos_hetero_n600_k8_cr.json"))
cat("wrote 2d_quantile_oos_hetero_n600_k8_cr.json\n")
