#!/usr/bin/env Rscript
# Median-regression reference on a response in raw dollars, via qgam.
#
#     Rscript scripts/r/gen_quantile_price_scale_fixture.R   (from the repo root)
#
# A median is equivariant to rescaling y, so a quantile fit on sale prices in
# dollars must give the same curve as the same fit in units of the median.
# The DGP mimics a small housing area: ~2 years of sale dates in days before
# today (t <= 0), a level of ~550k with a gentle seasonal wobble, and
# right-skewed (log-normal) noise. qgam's median at a few dates is the
# reference; tests/python/test_quantile_scale.py fits gamrs on y and on c*y.
suppressMessages({library(qgam); library(mgcv); library(jsonlite)})

SEED <- 20261022L
N    <- 600L
K    <- 10L
TAU  <- 0.5
T_EVAL <- c(-700, -540, -365, -180, -90, 0)

set.seed(SEED)
t <- sort(runif(N, -730, 0))
level <- 550000 * (1 + 0.03 * sin(2 * pi * t / 365) + 0.02 * t / 730)
y <- round(level * exp(0.25 * rnorm(N)), -3)
dat <- data.frame(t = t, y = y)

fit <- suppressMessages(qgam(y ~ s(t, k = K, bs = "cr"), data = dat, qu = TAU))
pred <- as.numeric(predict(fit, newdata = data.frame(t = T_EVAL)))
cat(sprintf("qgam median at t=%s: %s\n", paste(T_EVAL, collapse = ","),
            paste(round(pred), collapse = ",")))

obj <- list(
  schema_version = 1L,
  name = "1d_quantile_price_scale_n600_k10_cr",
  description = "1-D median regression on a raw-dollar response (qgam reference)",
  metadata = list(
    engine = "qgam",
    qgam_version = as.character(packageVersion("qgam")),
    mgcv_version = as.character(packageVersion("mgcv")),
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  ),
  inputs = list(seed = SEED, n = N, k = K, tau = TAU,
                x_train = t, y_train = y, x_eval = T_EVAL),
  qgam_output = list(pred_eval = pred)
)
writeLines(toJSON(obj, auto_unbox = TRUE, digits = 17),
           file.path("tests", "fixtures", "1d_quantile_price_scale_n600_k10_cr.json"))
