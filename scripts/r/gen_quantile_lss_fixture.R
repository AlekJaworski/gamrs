#!/usr/bin/env Rscript
# Distributional location-scale quantile reference via mgcv `gaulss`.
#
#     Rscript scripts/r/gen_quantile_lss_fixture.R
#
# gamrs's `fit_quantile_lss` is the distributional (location-scale) view of
# quantile regression: fit μ(x) and σ(x), then derive every quantile as
# q_τ(x) = μ(x) + σ(x)·Φ⁻¹(τ) (one fit → all τ, no crossing). The mgcv
# ground truth for exactly this model is the `gaulss` family — a JOINT
# Gaussian location-scale GAM with two linear predictors. gamrs estimates
# the same two functions in two stages (μ first, then σ on log|residual|);
# the parity bar is therefore OOS pinball / coverage, not an exact β match.
#
# DGP: 2-D, heteroskedastic IN x0 so the scale model is load-bearing —
# y = sin(2πx0) + 2(x1−½) + (0.1 + 0.4·x0)·N(0,1). Train + test split.
#
# mgcv gaulss `predict(type="response")` returns a 2-col matrix: column 1 is
# μ, column 2 is 1/σ (the precision). So σ = 1 / fitted[,2]. We verify that
# extraction by printing held-out coverage (should sit near τ) before writing.
suppressMessages({library(mgcv); library(jsonlite)})

FIX  <- "tests/fixtures"
SEED <- 20260610L
N_TR <- 800L
N_TE <- 400L
TAUS <- c(0.1, 0.5, 0.9)
K    <- 10L

pinball <- function(y, q, tau) {
  r <- y - q
  mean(pmax(tau * r, (tau - 1) * r))
}

gen <- function(n) {
  x0 <- runif(n)
  x1 <- runif(n)
  mu <- sin(2 * pi * x0) + 2 * (x1 - 0.5)
  sg <- 0.1 + 0.4 * x0
  list(x0 = x0, x1 = x1, y = mu + sg * rnorm(n))
}

set.seed(SEED)
tr <- gen(N_TR)
te <- gen(N_TE)
dat_tr <- data.frame(x0 = tr$x0, x1 = tr$x1, y = tr$y)
dat_te <- data.frame(x0 = te$x0, x1 = te$x1, y = te$y)

# gaulss: list(location_formula, scale_formula). Both smooth in (x0, x1).
fit <- gam(list(y ~ s(x0, k = 10, bs = "cr") + s(x1, k = 10, bs = "cr"),
                  ~ s(x0, k = 10, bs = "cr") + s(x1, k = 10, bs = "cr")),
           family = gaulss(), data = dat_tr)

resp   <- predict(fit, newdata = dat_te, type = "response")  # (n, 2): [μ, 1/σ]
mu_te  <- as.numeric(resp[, 1])
sig_te <- 1.0 / as.numeric(resp[, 2])

cat("gaulss 2-D location-scale quantile:\n")
per_tau <- list()
for (tau in TAUS) {
  q_te <- mu_te + sig_te * qnorm(tau)
  pb   <- pinball(dat_te$y, q_te, tau)
  cov  <- mean(dat_te$y <= q_te)
  per_tau[[as.character(tau)]] <- list(oos_pinball = pb, coverage = cov,
                                        pred_test = q_te)
  cat(sprintf("  tau=%.2f: gaulss OOS pinball=%.6f  coverage=%.3f\n", tau, pb, cov))
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
  name = "2d_quantile_lss_hetero_n800_k10_cr",
  description = "2-D distributional location-scale quantile reference (mgcv gaulss)",
  metadata = list(
    engine = "mgcv_gaulss",
    mgcv_version = as.character(packageVersion("mgcv")),
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  ),
  inputs = inputs,
  gaulss_output = list(per_tau = per_tau)
)
cat(toJSON(obj, auto_unbox = TRUE, digits = 17),
    file = file.path(FIX, "2d_quantile_lss_hetero_n800_k10_cr.json"))
cat("wrote 2d_quantile_lss_hetero_n800_k10_cr.json\n")
