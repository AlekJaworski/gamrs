#!/usr/bin/env Rscript
# Fellner-Schall (extended, `optimizer = "efs"`) reference fits for families
# with an estimated scale: 2-D additive Gamma(log), quasipoisson and
# quasibinomial (proportions with binomial sizes as prior weights), cr k=10.
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

emit <- function(path, desc, family, x0, x1, y, efs, reml, w = NULL) {
  n <- length(y)
  obj <- list(schema_version = 1L, name = sub("\\.json$", "", path), description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv"))),
              inputs = list(seed = 20261119L, n = n, d = 2L, k = c(k, k), bs = c("cr", "cr"),
                            family = family,
                            link = if (family == "quasibinomial") "logit" else "log",
                            x_train = lapply(seq_len(n), function(i) c(x0[i], x1[i])),
                            y_train = as.numeric(y)),
              efs = list(sp = as.numeric(efs$sp), scale = as.numeric(efs$scale),
                         predictions_train = as.numeric(fitted(efs)),
                         iter = as.integer(efs$outer.info$iter)),
              reml = list(sp = as.numeric(reml$sp), scale = as.numeric(reml$scale),
                          predictions_train = as.numeric(fitted(reml))))
  if (!is.null(w)) obj$inputs$weights <- w
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

size <- sample(5:30, n, replace = TRUE)
p <- plogis(0.8 * sin(2 * pi * x0) + 1.5 * cos(3 * x1) - 0.2)
# Overdispersed: a beta-binomial draw around p.
succ <- rbinom(n, size, rbeta(n, 10 * p, 10 * (1 - p)))
d <- data.frame(y = succ / size, w = size, x0 = x0, x1 = x1)
emit("2d_quasibinomial_logit_efs_n2000_k10_cr.json",
     "2-D additive quasibinomial (proportions, binomial sizes as weights), optimizer = efs",
     "quasibinomial", x0, x1, d$y,
     gam(f, family = quasibinomial, weights = w, data = d, optimizer = "efs", control = ctrl),
     gam(f, family = quasibinomial, weights = w, data = d, method = "REML", control = ctrl),
     w = as.numeric(size))
