#!/usr/bin/env Rscript
# Multi-smooth (2- and 3-margin) scaled-t (`scat`) parity fixtures.
#
#     Rscript scripts/r/gen_scat_multismooth_fixtures.R
#
# Companion to gen_multismooth_fixtures.R. scat is the hardest convergence
# regime in the battery (joint Newton over [log λ…, log σ², log(ν−min.df)]),
# and the README flagged multi-smooth scat reference parity as "pending" —
# this closes that gap. Data uses heavy-tailed t(df=4) noise so the robust
# down-weighting is load-bearing (a Gaussian fit would chase the tails).
#
# Output mirrors gen_multismooth_fixtures.R's emit() exactly. scat uses the
# identity link, so mgcv predict(type="response") == the η/μ scale gamrs
# predicts. Extra field: scat_theta = fit$family$getTheta(TRUE) (the fitted
# (ν, σ) pair, on mgcv's convention).
suppressMessages({library(mgcv); library(jsonlite)})

FIX <- "tests/fixtures"

emit <- function(path, name, desc, d, ks, x_list, y, fit, extra = list()) {
  n <- length(y)
  pred <- as.numeric(predict(fit, type = "response"))
  inputs <- list(seed = 20260609L, n = n, d = d, k = ks,
                 bs = rep("cr", d), family = "scat", link = "identity",
                 method = "REML", weights = NULL,
                 x_train = lapply(seq_len(n),
                                  function(i) sapply(x_list, function(col) col[i])),
                 y_train = as.numeric(y))
  inputs <- c(inputs, extra)
  mo <- list(predictions_train = pred, scale = as.numeric(fit$scale),
             edf_total = as.numeric(sum(fit$edf)), beta = as.numeric(coef(fit)))
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv")),
                              generated_at = format(Sys.time(),
                                                    "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
              inputs = inputs, mgcv_output = mo)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  th <- as.numeric(fit$family$getTheta(TRUE))
  cat(sprintf("  %-40s edf=%.3f  theta=(%s)\n", name, sum(fit$edf),
              paste(sprintf("%.3f", th), collapse = ", ")))
}

# ---- 2-D additive scat ----------------------------------------------------
cat("scat 2-D:\n")
set.seed(20260609)
n <- 600
x0 <- runif(n); x1 <- runif(n)
f  <- 3 + sin(2 * pi * x0) + 0.7 * (x1 - 0.5)
y2 <- f + rt(n, df = 4) * 0.5
fit2 <- gam(y2 ~ s(x0, k = 8, bs = "cr") + s(x1, k = 8, bs = "cr"),
            family = scat(), method = "REML")
emit("2d_scat_identity_n600_k8_cr.json", "2d_scat_identity_n600_k8_cr",
     "2-D additive scat (mgcv scat(), identity), cr k=8 margins, t(4) noise",
     2L, c(8L, 8L), list(x0, x1), y2, fit2,
     list(scat_theta = as.numeric(fit2$family$getTheta(TRUE))))

# ---- 3-D additive scat ----------------------------------------------------
cat("scat 3-D:\n")
set.seed(20260610)
n <- 800
x0 <- runif(n); x1 <- runif(n); x2 <- runif(n)
f  <- 2.5 + sin(2 * pi * x0) + 0.6 * x1 + 0.5 * (x2 - 0.5)^2
y3 <- f + rt(n, df = 5) * 0.4
fit3 <- gam(y3 ~ s(x0, k = 8, bs = "cr") + s(x1, k = 8, bs = "cr")
              + s(x2, k = 8, bs = "cr"),
            family = scat(), method = "REML")
emit("3d_scat_identity_n800_k8_cr.json", "3d_scat_identity_n800_k8_cr",
     "3-D additive scat (mgcv scat(), identity), cr k=8 margins, t(5) noise",
     3L, c(8L, 8L, 8L), list(x0, x1, x2), y3, fit3,
     list(scat_theta = as.numeric(fit3$family$getTheta(TRUE))))
