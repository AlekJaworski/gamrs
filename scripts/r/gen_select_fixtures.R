#!/usr/bin/env Rscript
# Parity fixtures for gam(select = TRUE), Gaussian REML:
#   y ~ s(x1) + s(x2) + s(x3), cr, with x3 carrying no signal;
#   y ~ te(x, z, k = c(4, 3), bs = "cr");
#   y ~ te(x, z, k = c(4, 3), bs = "cr", by = w) + s(v), the tensor carrying no signal.
# Data is embedded so the Rust and Python tests fit exactly what mgcv saw.
# Run from the repo root.
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"

emit <- function(name, desc, formula, train, new) {
  fit <- gam(formula, method = "REML", data = train, select = TRUE,
             control = gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-10)))
  cols <- setdiff(names(train), "y")
  terms_train <- predict(fit, type = "terms")
  terms_new <- predict(fit, newdata = new, type = "terms")
  blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
  as_rows <- function(d) lapply(seq_len(nrow(d)), function(i) as.numeric(d[i, cols]))
  inputs <- list(n = nrow(train), columns = cols, method = "REML",
                 x_train = as_rows(train), y_train = as.numeric(train$y),
                 x_new = as_rows(new))
  mo <- list(fitted = as.numeric(fitted(fit)),
             term_train = lapply(seq_len(ncol(terms_train)), function(j) as.numeric(terms_train[, j])),
             term_new = lapply(seq_len(ncol(terms_new)), function(j) as.numeric(terms_new[, j])),
             predictions_new = as.numeric(predict(fit, newdata = new)),
             edf_per_term = I(sapply(blocks, function(b) sum(fit$edf[b]))),
             n_coef_per_term = I(sapply(blocks, length)),
             reml = as.numeric(fit$gcv.ubre),
             scale = as.numeric(fit$sig2),
             sp = I(as.numeric(fit$sp)))
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv")),
                              formula = deparse(formula), select = TRUE),
              inputs = inputs, mgcv_output = mo)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, paste0(name, ".json")))
  cat(sprintf("  %-36s edf=(%s) reml=%.6f sp=(%s)\n", name,
              paste(sprintf("%.3f", mo$edf_per_term), collapse = ", "), mo$reml,
              paste(sprintf("%.4g", mo$sp), collapse = ", ")))
}

set.seed(20261123)
n <- 400

x1 <- runif(n); x2 <- runif(n); x3 <- runif(n)
y <- sin(2 * pi * x1) + 0.6 * x2 + rnorm(n, sd = 0.4)
new <- data.frame(x1 = seq(0.02, 0.98, length.out = 20), x2 = seq(0.9, 0.1, length.out = 20),
                  x3 = (seq_len(20) * 0.37) %% 1)
emit("select_cr_gaussian_n400",
     "s(x1) + s(x2) + s(x3), cr k = 10, 8, 8, x2 linear and x3 pure noise, Gaussian REML, select = TRUE",
     y ~ s(x1, k = 10, bs = "cr") + s(x2, k = 8, bs = "cr") + s(x3, k = 8, bs = "cr"),
     data.frame(x1 = x1, x2 = x2, x3 = x3, y = y), new)

x <- runif(n); z <- runif(n)
y <- sin(2 * pi * x) * (1 + z) + rnorm(n, sd = 0.3)
x_new <- seq(0.02, 0.98, length.out = 24)
emit("select_te_gaussian_n400_k4x3",
     "te(x, z), cr k = (4, 3), Gaussian REML, select = TRUE",
     y ~ te(x, z, k = c(4, 3), bs = "cr"),
     data.frame(x = x, z = z, y = y), data.frame(x = x_new, z = rev(x_new)))

x <- runif(n, 0, 10); z <- runif(n, 0, 450); v <- runif(n)
w <- log1p(5 * rexp(n)) * (1 - z / 450)
y <- 3 * sin(2 * pi * v) + rnorm(n, sd = 2)
new <- data.frame(x = seq(0.2, 9.8, length.out = 24), z = seq(440, 10, length.out = 24),
                  w = rep(c(1.3, 0, 0.4, 2), length.out = 24), v = (seq_len(24) * 0.29) %% 1)
emit("select_te_by_null_gaussian_n400_k4x3",
     "te(x, z, by = w) + s(v), cr k = (4, 3) and 8, the tensor carrying no signal, Gaussian REML, select = TRUE",
     y ~ te(x, z, k = c(4, 3), bs = "cr", by = w) + s(v, k = 8, bs = "cr"),
     data.frame(x = x, z = z, w = w, v = v, y = y), new)
