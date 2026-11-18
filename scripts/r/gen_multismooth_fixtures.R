#!/usr/bin/env Rscript
# Generate 2-D additive parity fixtures for the multi-smooth NB + Tweedie
# (profile-p and fixed-p) paths. Data is embedded in each fixture so the Rust
# and Python parity tests fit the identical (x, y) mgcv saw.
suppressMessages({library(mgcv); library(MASS); library(jsonlite)})

FIX <- "tests/fixtures"
k0 <- 8L; k1 <- 8L

emit <- function(path, name, desc, family, link, x0, x1, y, fit, extra = list()) {
  n <- length(y)
  pred <- as.numeric(predict(fit, type = "response"))
  inputs <- list(seed = 20260529L, n = n, d = 2L, k = c(k0, k1),
                 bs = c("cr", "cr"), family = family, link = link,
                 method = "REML", weights = NULL,
                 x_train = lapply(seq_len(n), function(i) c(x0[i], x1[i])),
                 y_train = as.numeric(y))
  inputs <- c(inputs, extra)
  mo <- list(predictions_train = pred, scale = as.numeric(fit$scale),
             edf_total = as.numeric(sum(fit$edf)), beta = as.numeric(coef(fit)))
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv")),
                              generated_at = "2026-05-29T00:00:00Z"),
              inputs = inputs, mgcv_output = mo)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  cat(sprintf("  %-38s edf=%.3f scale=%.4f\n", name, sum(fit$edf), fit$scale))
}

# ---- Tweedie (shared data; profile-p and fixed-p fit the same y) ----------
set.seed(20260529)
n <- 600
x0 <- runif(n); x1 <- runif(n)
eta <- 0.7 + sin(2 * pi * x0) + 0.6 * (x1 - 0.5)
mu  <- exp(eta)
y_tw <- rTweedie(mu, p = 1.5, phi = 1.0)
dat_tw <- data.frame(y = y_tw, x0 = x0, x1 = x1)

cat("Tweedie:\n")
fit_prof <- gam(y ~ s(x0, k = k0, bs = "cr") + s(x1, k = k1, bs = "cr"),
                family = tw(), method = "REML", data = dat_tw)
p_hat <- as.numeric(fit_prof$family$getTheta(TRUE))
emit("2d_tw_profile_log_n600_k8_cr.json", "2d_tw_profile_log_n600_k8_cr",
     "2-D additive Tweedie, profile-p (mgcv tw()), cr k=8 margins",
     "tw", "log", x0, x1, y_tw, fit_prof, list(tweedie_p_hat = p_hat))

fit_fix <- gam(y ~ s(x0, k = k0, bs = "cr") + s(x1, k = k1, bs = "cr"),
               family = Tweedie(p = 1.5, link = "log"), method = "REML", data = dat_tw)
emit("2d_tw_fixed_p15_log_n600_k8_cr.json", "2d_tw_fixed_p15_log_n600_k8_cr",
     "2-D additive Tweedie, FIXED p=1.5 (mgcv Tweedie(p=1.5)), cr k=8 margins",
     "tw", "log", x0, x1, y_tw, fit_fix, list(tweedie_p = 1.5))

# ---- NegBin (profile-theta, mgcv nb()) ------------------------------------
cat("NegBin:\n")
set.seed(20260530)
x0 <- runif(n); x1 <- runif(n)
eta <- 0.5 + 0.9 * sin(2 * pi * x0) + 0.7 * x1
mu  <- exp(eta)
theta_true <- 4.0
y_nb <- rnegbin(n, mu = mu, theta = theta_true)
dat_nb <- data.frame(y = y_nb, x0 = x0, x1 = x1)
fit_nb <- gam(y ~ s(x0, k = k0, bs = "cr") + s(x1, k = k1, bs = "cr"),
              family = nb(), method = "REML", data = dat_nb)
theta_hat <- as.numeric(fit_nb$family$getTheta(TRUE))
emit("2d_nb_log_n600_k8_cr.json", "2d_nb_log_n600_k8_cr",
     "2-D additive NegBin, profile-theta (mgcv nb()), cr k=8 margins",
     "nb", "log", x0, x1, y_nb, fit_nb, list(nb_theta_hat = theta_hat))
cat(sprintf("  (nb theta_hat=%.4f)\n", theta_hat))
