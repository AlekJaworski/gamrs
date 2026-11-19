#!/usr/bin/env Rscript
# bam(method = "fREML") reference fits: 3-D additive cr k=10 for gaussian
# (unweighted and prior-weighted), nb(log) with estimated theta, and
# scat(identity) with prior weights, and Gamma(log) for the estimated-scale
# GLM branch. n stays under bam's chunk.size, so the
# basis is built from every row. x2 enters linearly, so its smooth sits on a
# flat REML ridge (lambda -> large) where the stopping rule decides lambda.
#
#     Rscript scripts/r/gen_bam_freml_fixtures.R
#
# Each fixture records the bam fit and, for comparison, gam(method = "REML").
suppressMessages({library(mgcv); library(jsonlite)})

FIX <- "tests/fixtures"
k <- 10L
f3 <- y ~ s(x0, k = k, bs = "cr") + s(x1, k = k, bs = "cr") + s(x2, k = k, bs = "cr")

summarise <- function(fit) {
  edf_term <- vapply(fit$smooth, function(s) sum(fit$edf[s$first.para:s$last.para]), 0)
  out <- list(sp = as.numeric(fit$sp), scale = as.numeric(fit$sig2),
              predictions_train = as.numeric(fitted(fit)),
              linear_predictors = as.numeric(fit$linear.predictors),
              beta = as.numeric(coef(fit)), edf_per_term = as.numeric(edf_term),
              edf_total = sum(fit$edf), reml = as.numeric(fit$gcv.ubre),
              iter = as.integer(fit$outer.info$iter), irls_iter = as.integer(fit$iter))
  th <- fit$family$getTheta
  if (!is.null(th)) out$theta <- I(as.numeric(th(TRUE)))
  out
}

emit <- function(path, desc, family, link, d, bam_fit, gam_fit, w = NULL) {
  n <- nrow(d)
  inputs <- list(seed = 20261119L, n = n, d = 3L, k = rep(k, 3), bs = rep("cr", 3),
                 family = family, link = link,
                 x_train = lapply(seq_len(n), function(i) c(d$x0[i], d$x1[i], d$x2[i])),
                 y_train = as.numeric(d$y))
  if (!is.null(w)) inputs$weights <- as.numeric(w)
  name <- sub("\\.json$", "", path)
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv"))),
              inputs = inputs, bam = summarise(bam_fit), gam_reml = summarise(gam_fit))
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  cat(sprintf("  %-44s bam edf=%.4f log sp %s (iter %d) | gam edf=%.4f log sp %s\n", name,
              sum(bam_fit$edf), paste(round(log(bam_fit$sp), 4), collapse = ","),
              bam_fit$outer.info$iter, sum(gam_fit$edf),
              paste(round(log(gam_fit$sp), 4), collapse = ",")))
}

set.seed(20261119)
n <- 2000
x0 <- runif(n); x1 <- runif(n); x2 <- runif(n)
eta <- 0.8 * sin(2 * pi * x0) + 0.6 * cos(3 * x1) + 0.4 * x2

d <- data.frame(y = 2 + eta + rnorm(n, sd = 0.4), x0 = x0, x1 = x1, x2 = x2)
emit("3d_gaussian_bam_freml_n2000_k10_cr.json", "3-D additive gaussian, bam fREML",
     "gaussian", "identity", d,
     bam(f3, data = d, method = "fREML"), gam(f3, data = d, method = "REML"))

w <- runif(n, 0.3, 3)
d <- data.frame(y = 2 + eta + rnorm(n, sd = 0.4 / sqrt(w)), x0 = x0, x1 = x1, x2 = x2, w = w)
emit("3d_gaussian_weighted_bam_freml_n2000_k10_cr.json",
     "3-D additive gaussian, prior weights U(0.3, 3), bam fREML", "gaussian", "identity", d,
     bam(f3, data = d, weights = w, method = "fREML"),
     gam(f3, data = d, weights = w, method = "REML"), w = w)

d <- data.frame(y = rnbinom(n, size = 3, mu = exp(1.5 + eta)), x0 = x0, x1 = x1, x2 = x2)
emit("3d_nb_log_bam_freml_n2000_k10_cr.json", "3-D additive nb(log), theta estimated, bam fREML",
     "negbin", "log", d,
     bam(f3, data = d, family = nb(), method = "fREML"),
     gam(f3, data = d, family = nb(), method = "REML"))

n <- 1500
x0 <- x0[1:n]; x1 <- x1[1:n]; x2 <- x2[1:n]
w <- runif(n, 0.5, 2)
y <- 10 + 3 * (0.8 * sin(2 * pi * x0) + 0.6 * cos(3 * x1) + 0.4 * x2) + 1.5 * rt(n, df = 5)
d <- data.frame(y = y, x0 = x0, x1 = x1, x2 = x2, w = w)
emit("3d_scat_identity_weighted_bam_freml_n1500_k10_cr.json",
     "3-D additive scat(identity), prior weights U(0.5, 2), bam fREML", "scat", "identity", d,
     bam(f3, data = d, weights = w, family = scat(link = "identity"), method = "fREML"),
     gam(f3, data = d, weights = w, family = scat(link = "identity"), method = "REML"), w = w)

set.seed(20261120)
n <- 2000
x0 <- runif(n); x1 <- runif(n); x2 <- runif(n)
mu <- exp(0.5 + 0.8 * sin(2 * pi * x0) + 0.6 * cos(3 * x1) + 0.4 * x2)
d <- data.frame(y = rgamma(n, shape = 3, rate = 3 / mu), x0 = x0, x1 = x1, x2 = x2)
emit("3d_gamma_log_bam_freml_n2000_k10_cr.json", "3-D additive Gamma(log), bam fREML",
     "gamma", "log", d,
     bam(f3, data = d, family = Gamma(link = "log"), method = "fREML"),
     gam(f3, data = d, family = Gamma(link = "log"), method = "REML"))

# Above chunk.size (10,000) bam builds its bases from a seeded row sample.
# These inputs are an integer lattice, not drawn from R's RNG, so the tests
# rebuild them bit for bit from n alone; only outputs are stored (fitted
# values at every 100th row).
lattice <- function(n) {
  i <- seq_len(n)
  u <- function(a, m) ((i * a) %% m) / m
  x0 <- u(7919, 10007); x1 <- u(6037, 10009); x2 <- u(4001, 10037)
  e1 <- u(7907, 10039) - 0.5; e2 <- u(5003, 10061) - 0.5; e3 <- u(3001, 10067)
  s <- 4 * x0 * (1 - x0) - 2 * (x1 - 0.5) * (x1 - 0.5) + 0.4 * x2
  data.frame(x0 = x0, x1 = x1, x2 = x2,
             yg = 2 + s + 0.8 * e1,
             ynb = floor((1.5 + 3 * s) * 2 * (0.5 + e1) * (0.5 + e2) * 2),
             ys = 10 + 3 * s + 0.3 * e1 / (0.05 + abs(e2)),
             w = 0.5 + 1.5 * e3)
}
emit_large <- function(path, desc, family, d, fit) {
  every <- seq(100, nrow(d), by = 100)
  out <- summarise(fit)
  out$predictions_train <- NULL; out$linear_predictors <- NULL; out$beta <- NULL
  out$rows <- every; out$predictions_rows <- as.numeric(fitted(fit))[every]
  obj <- list(schema_version = 1L, name = sub("\\.json$", "", path), description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv"))),
              inputs = list(n = nrow(d), k = rep(k, 3), bs = rep("cr", 3), family = family,
                            lattice = "scripts/r/gen_bam_freml_fixtures.R lattice()"),
              bam = out)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  cat(sprintf("  %-46s edf=%.4f log sp %s iter %d irls %d\n", obj$name, sum(fit$edf),
              paste(round(log(fit$sp), 4), collapse = ","), fit$outer.info$iter, fit$iter))
}
fx <- function(y) as.formula(sprintf("%s ~ s(x0, k = %d, bs = 'cr') + s(x1, k = %d, bs = 'cr') + s(x2, k = %d, bs = 'cr')", y, k, k, k))
for (n in c(12000L, 25000L)) {
  d <- lattice(n)
  emit_large(sprintf("3d_gaussian_bam_freml_n%d_lattice.json", n), "lattice gaussian, default chunk.size",
             "gaussian", d, bam(fx("yg"), data = d, method = "fREML"))
  emit_large(sprintf("3d_nb_log_bam_freml_n%d_lattice.json", n), "lattice nb(log), default chunk.size",
             "negbin", d, bam(fx("ynb"), data = d, family = nb(), method = "fREML"))
  emit_large(sprintf("3d_scat_identity_weighted_bam_freml_n%d_lattice.json", n),
             "lattice scat(identity), prior weights, default chunk.size", "scat", d,
             bam(fx("ys"), data = d, weights = w, family = scat(link = "identity"), method = "fREML"))
}
