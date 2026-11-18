#!/usr/bin/env Rscript
# Prior-weighted parity fixtures: 2-D additive Gamma, inverse Gaussian, NegBin
# and Tweedie (log link, cr k=8 margins, REML) with non-unit prior weights.
#
#     Rscript scripts/r/gen_weighted_fixtures.R
#
# Same schema as gen_multismooth_fixtures.R plus `inputs.weights` and
# `mgcv_output.{sp,reml}`; NegBin adds `nb_theta_hat`, Tweedie
# `tweedie_p_hat`.
suppressMessages({library(mgcv); library(MASS); library(jsonlite)})

FIX <- "tests/fixtures"
k <- 8L

emit <- function(path, desc, family, x0, x1, y, w, fit, extra = list()) {
  n <- length(y)
  inputs <- list(seed = 20261118L, n = n, d = 2L, k = c(k, k),
                 bs = c("cr", "cr"), family = family, link = "log",
                 method = "REML", weights = as.numeric(w),
                 x_train = lapply(seq_len(n), function(i) c(x0[i], x1[i])),
                 y_train = as.numeric(y))
  inputs <- c(inputs, extra)
  mo <- list(predictions_train = as.numeric(predict(fit, type = "response")),
             scale = as.numeric(fit$scale),
             edf_total = as.numeric(sum(fit$edf)),
             sp = as.numeric(fit$sp),
             reml = as.numeric(fit$gcv.ubre),
             beta = as.numeric(coef(fit)))
  name <- sub("\\.json$", "", path)
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv"))),
              inputs = inputs, mgcv_output = mo)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  cat(sprintf("  %-40s edf=%.3f scale=%.4f sp=%s\n", name, sum(fit$edf),
              fit$scale, paste(signif(fit$sp, 5), collapse = ",")))
}

fit2 <- function(fam, dat) {
  gam(y ~ s(x0, k = k, bs = "cr") + s(x1, k = k, bs = "cr"), family = fam,
      weights = w, method = "REML", data = dat,
      control = gam.control(epsilon = 1e-10))
}

set.seed(20261118)
n <- 400
x0 <- runif(n); x1 <- runif(n)
w <- runif(n, 0.2, 3)
mu <- exp(0.5 + 0.7 * sin(2 * pi * x0) + 0.5 * (x1 - 0.5))

y <- rgamma(n, shape = 2 * w, rate = 2 * w / mu)
d <- data.frame(y = y, x0 = x0, x1 = x1, w = w)
emit("2d_gamma_log_weighted_n400_k8_cr.json",
     "2-D additive Gamma(log), prior weights U(0.2, 3)", "gamma", x0, x1, y, w,
     fit2(Gamma(link = "log"), d))

rinvgauss <- function(mu, lambda) {
  nu <- rnorm(length(mu))^2
  x <- mu + mu^2 * nu / (2 * lambda) -
    mu / (2 * lambda) * sqrt(4 * mu * lambda * nu + mu^2 * nu^2)
  ifelse(runif(length(mu)) <= mu / (mu + x), x, mu^2 / x)
}
y <- rinvgauss(mu, w / 0.3)
d <- data.frame(y = y, x0 = x0, x1 = x1, w = w)
emit("2d_invgauss_log_weighted_n400_k8_cr.json",
     "2-D additive inverse Gaussian(log), prior weights U(0.2, 3)", "invgauss",
     x0, x1, y, w, fit2(inverse.gaussian(link = "log"), d))

y <- rnegbin(n, mu = mu, theta = 3)
d <- data.frame(y = y, x0 = x0, x1 = x1, w = w)
f <- fit2(nb(), d)
emit("2d_nb_log_weighted_n400_k8_cr.json",
     "2-D additive NegBin profile-theta, prior weights U(0.2, 3)", "nb",
     x0, x1, y, w, f, list(nb_theta_hat = as.numeric(f$family$getTheta(TRUE))))

y <- rTweedie(mu, p = 1.5, phi = 1)
d <- data.frame(y = y, x0 = x0, x1 = x1, w = w)
f <- fit2(tw(), d)
emit("2d_tw_log_weighted_n400_k8_cr.json",
     "2-D additive Tweedie profile-p, prior weights U(0.2, 3)", "tw",
     x0, x1, y, w, f, list(tweedie_p_hat = as.numeric(f$family$getTheta(TRUE))))
