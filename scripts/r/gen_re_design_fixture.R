#!/usr/bin/env Rscript
# Parity fixture for a predictors-only random-effect model, Gaussian REML:
#   y ~ s(g1, bs = "re") + s(g2, bs = "re")
# with g1 (12 levels) and g2 (7 levels) stored as integer codes. Run from the
# repo root.
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"

set.seed(20261120)
n <- 300
g1 <- sample(0:11, n, replace = TRUE)
g2 <- sample(0:6, n, replace = TRUE)
y <- 1 + rnorm(12, sd = 0.8)[g1 + 1] + rnorm(7, sd = 0.4)[g2 + 1] + rnorm(n, sd = 0.5)
d <- data.frame(g1 = factor(g1), g2 = factor(g2), y = y)
fit <- gam(y ~ s(g1, bs = "re") + s(g2, bs = "re"), data = d, method = "REML",
           control = gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-10)))

out <- list(
  name = "re_design_gaussian_n300",
  description = "y ~ s(g1, bs='re') + s(g2, bs='re'), gaussian REML; g1, g2 integer codes",
  inputs = list(n = n, columns = c("g1", "g2"),
                x_train = lapply(seq_len(n), function(i) c(g1[i], g2[i])),
                y_train = y),
  mgcv_output = list(fitted = as.numeric(fitted(fit)), sp = as.numeric(fit$sp),
                     scale = fit$scale, edf_total = sum(fit$edf),
                     reml = as.numeric(fit$gcv.ubre),
                     mgcv_version = as.character(packageVersion("mgcv")))
)
write_json(out, file.path(FIX, "re_design_gaussian_n300.json"), digits = NA, auto_unbox = TRUE)
