#!/usr/bin/env Rscript
# R's Mersenne-Twister + sample() draws, and the rows mgcv's mini.mf (bam's
# basis-setup subsample above chunk.size) picks, for the r_rng unit tests.
# mini.mf overwrites rows in place, so its rows are identified by their
# (distinct) x0 values rather than by row names.
#
#     Rscript scripts/r/gen_r_rng_fixture.R
suppressMessages({library(mgcv); library(jsonlite)})
RNGkind("default", "default", "Rejection")
draws <- list()
for (case in list(c(66, 10, 3), c(66, 12000, 10000), c(1, 1000, 1000), c(12345, 100000, 25), c(7, 2^20 + 3, 50))) {
  set.seed(case[1]); u <- round(runif(5) * 2^32)
  set.seed(case[1]); s <- sample(case[2], case[3])
  draws[[length(draws) + 1]] <- list(seed = case[1], n = case[2], size = case[3], unif_2_32 = u, sample = s)
}
set.seed(20261119)
n <- 600
mf <- data.frame(y = rpois(n, 3), x0 = runif(n), g = factor(sample(letters[1:7], n, TRUE)),
                 x1 = rnorm(n), "(weights)" = runif(n), check.names = FALSE)
mf0 <- mgcv:::mini.mf(mf, 200)
obj <- list(r_version = R.version.string, draws = draws,
            mini_mf = list(y = mf$y, x0 = mf$x0, g = as.integer(mf$g), x1 = mf$x1, w = mf[["(weights)"]],
                           chunk_size = 200L, x0_of_rows = mf0$x0))
cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = "tests/fixtures/r_rng_and_mini_mf.json")
