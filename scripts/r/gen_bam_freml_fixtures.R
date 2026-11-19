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
