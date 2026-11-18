#!/usr/bin/env Rscript
# Fellner-Schall (extended, `optimizer = "efs"`) reference fits for families
# with an estimated scale: 2-D additive Gamma(log) and quasipoisson, cr k=10.
#
#     Rscript scripts/r/gen_efs_fixtures.R
#
# Records the efs smoothing parameters and fitted values, plus the REML fit's
# for comparison. efs.tol is tightened so the reference sits at the update's
# fixed point rather than mgcv's default early stop.
suppressMessages({library(mgcv); library(jsonlite)})

FIX <- "tests/fixtures"
k <- 10L
ctrl <- gam.control(efs.tol = 1e-10, epsilon = 1e-10, maxit = 200)

emit <- function(path, desc, family, x0, x1, y, efs, reml) {
  n <- length(y)
  obj <- list(schema_version = 1L, name = sub("\\.json$", "", path), description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv"))),
              inputs = list(seed = 20261119L, n = n, d = 2L, k = c(k, k), bs = c("cr", "cr"),
                            family = family, link = "log",
                            x_train = lapply(seq_len(n), function(i) c(x0[i], x1[i])),
                            y_train = as.numeric(y)),
              efs = list(sp = as.numeric(efs$sp), scale = as.numeric(efs$scale),
                         predictions_train = as.numeric(fitted(efs)),
                         iter = as.integer(efs$outer.info$iter)),
              reml = list(sp = as.numeric(reml$sp), scale = as.numeric(reml$scale),
                          predictions_train = as.numeric(fitted(reml))))
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  cat(sprintf("  %-34s efs log sp %s (iter %d) | REML log sp %s\n", obj$name,
              paste(round(log(efs$sp), 4), collapse = ","), efs$outer.info$iter,
              paste(round(log(reml$sp), 4), collapse = ",")))
}

set.seed(20261119)
n <- 2000
x0 <- runif(n); x1 <- runif(n)
mu <- exp(0.5 + 0.8 * sin(2 * pi * x0) + 0.6 * cos(3 * x1))
f <- y ~ s(x0, k = k, bs = "cr") + s(x1, k = k, bs = "cr")

d <- data.frame(y = rgamma(n, shape = 3, rate = 3 / mu), x0 = x0, x1 = x1)
emit("2d_gamma_log_efs_n2000_k10_cr.json", "2-D additive Gamma(log), optimizer = efs",
     "gamma", x0, x1, d$y,
     gam(f, family = Gamma(link = "log"), data = d, optimizer = "efs", control = ctrl),
     gam(f, family = Gamma(link = "log"), data = d, method = "REML", control = ctrl))

d <- data.frame(y = rnbinom(n, size = 2, mu = mu), x0 = x0, x1 = x1)
emit("2d_quasipoisson_log_efs_n2000_k10_cr.json", "2-D additive quasipoisson, optimizer = efs",
     "quasipoisson", x0, x1, d$y,
     gam(f, family = quasipoisson, data = d, optimizer = "efs", control = ctrl),
     gam(f, family = quasipoisson, data = d, method = "REML", control = ctrl))
