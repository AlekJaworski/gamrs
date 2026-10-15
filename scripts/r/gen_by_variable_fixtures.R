#!/usr/bin/env Rscript
# Parity fixtures for numeric `by=` smooths:
#   y ~ s(x, k = 8, bs = "cr", by = w) + s(z, k = 8, bs = "cr"), method = "REML"
# once with w a 0/1 indicator and once with w a step (1, 0.6, 0.25, 0), the
# traffic-curve-times-distance-band shape. Data is embedded so the Rust and
# Python tests fit exactly what mgcv saw. Run from the repo root.
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"
K <- 8L

emit <- function(path, name, desc, x, z, w, y, x_new, z_new, w_new) {
  dat <- data.frame(y = y, x = x, z = z, w = w)
  fit <- gam(y ~ s(x, k = K, bs = "cr", by = w) + s(z, k = K, bs = "cr"),
             method = "REML", data = dat)
  new <- data.frame(x = x_new, z = z_new, w = w_new)
  terms_train <- predict(fit, type = "terms")
  terms_new <- predict(fit, newdata = new, type = "terms")
  n <- length(y)
  blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
  inputs <- list(n = n, k = c(K, K), method = "REML",
                 x_train = lapply(seq_len(n), function(i) c(x[i], z[i], w[i])),
                 y_train = as.numeric(y),
                 x_new = lapply(seq_along(x_new), function(i) c(x_new[i], z_new[i], w_new[i])))
  mo <- list(fitted = as.numeric(fitted(fit)),
             coef_by_term = as.numeric(coef(fit)[blocks[[1]]]),
             term_train = list(as.numeric(terms_train[, 1]), as.numeric(terms_train[, 2])),
             term_new = list(as.numeric(terms_new[, 1]), as.numeric(terms_new[, 2])),
             predictions_new = as.numeric(predict(fit, newdata = new)),
             intercept = as.numeric(coef(fit)[1]),
             edf_per_term = sapply(blocks, function(b) sum(fit$edf[b])),
             edf_total = sum(fit$edf),
             reml = as.numeric(fit$gcv.ubre),
             scale = as.numeric(fit$sig2),
             sp = as.numeric(fit$sp),
             n_coef_by_term = length(blocks[[1]]))
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv"))),
              inputs = inputs, mgcv_output = mo)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  cat(sprintf("  %-34s edf=(%.3f, %.3f) reml=%.6f sp=(%.4g, %.4g)\n", name,
              mo$edf_per_term[1], mo$edf_per_term[2], mo$reml, mo$sp[1], mo$sp[2]))
}

set.seed(20261015)
n <- 400
x <- runif(n); z <- runif(n)
x_new <- seq(0.02, 0.98, length.out = 25); z_new <- rev(x_new)

w_ind <- rbinom(n, 1, 0.5)
y_ind <- 1 + w_ind * 1.5 * sin(2 * pi * x) + 0.5 * cos(pi * z) + rnorm(n, sd = 0.3)
emit("by_indicator_gaussian_n400_k8_cr.json", "by_indicator_gaussian_n400_k8_cr",
     "s(x, by = w) + s(z), w a 0/1 indicator, Gaussian REML, cr k = 8",
     x, z, w_ind, y_ind, x_new, z_new, rep(c(1, 0), length.out = 25))

bands <- c(1, 0.6, 0.25, 0)
w_step <- bands[sample.int(4, n, replace = TRUE)]
y_step <- 1 + w_step * (2 * x + sin(3 * pi * x)) + 0.5 * cos(pi * z) + rnorm(n, sd = 0.3)
emit("by_step_gaussian_n400_k8_cr.json", "by_step_gaussian_n400_k8_cr",
     "s(x, by = w) + s(z), w a distance-band step (1, 0.6, 0.25, 0), Gaussian REML, cr k = 8",
     x, z, w_step, y_step, x_new, z_new, rep(bands, length.out = 25))
