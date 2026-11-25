# Gaussian REML search-path fixtures.
#
#   Rscript scripts/r/gen_search_path_fixtures.R
#
# REML over several smooths can have more than one local minimum, and which
# one a search reaches depends on its whole path. These synthetic fixtures
# (no real data) pin gamrs's Gaussian search to mgcv's:
#
#   tests/fixtures/strict/gaussian_start_basin_n300.json
#     Six cr smooths on raw-scale, pricing-like covariates (seed 45). mgcv from
#     its own start reaches the lower of two minima; started from sp = 1 it
#     stops in the higher one (`from_sp1`). Needs mgcv's start and its joint
#     (rho, log phi) search.
#   tests/fixtures/strict/gaussian_indefinite_step_n320.json
#     Nine cr smooths + two linear terms shaped like a home-pricing GAM
#     (seed 180). The first steps meet an indefinite Hessian; only mgcv's
#     handling of it (a steepest-descent search beside the Newton step)
#     reaches mgcv's minimum. A lower minimum exists (`lower_minimum`), which
#     mgcv's search does not find: the fixture pins parity, not the global
#     optimum.
#
# Each holds mgcv's strict fit (control as in gen_strict_fixtures.R) and its
# outer path: (log sp, log scale) at the start of every newton() iteration,
# recorded through newton()'s own sp.trace hook.
suppressMessages({library(mgcv); library(jsonlite)})
ctl <- gam.control(epsilon=1e-12, maxit=400, newton=list(conv.tol=1e-12, maxNstep=5, maxHalf=60))

# newton() with its sp.trace hook recording into .path instead of printing.
.path <- list()
src <- deparse(mgcv:::newton)
src <- sub("sp.trace <- FALSE", "sp.trace <- TRUE", src, fixed = TRUE)
src <- sub('cat(lsp, "\\n")', ".path[[length(.path) + 1]] <<- drop(lsp)", src, fixed = TRUE)
stopifnot(any(grepl(".path[[", src, fixed = TRUE)))
nt <- eval(parse(text = src)); environment(nt) <- asNamespace("mgcv")
assignInNamespace("newton", nt, "mgcv")

fit_with_path <- function(fo, d, ...) {
  .path <<- list()
  m <- gam(fo, data = d, method = "REML", control = ctl, ...)
  list(m = m, path = do.call(rbind, .path))
}

write_fixture <- function(name, d, cols, k, fo, extra = list()) {
  r <- fit_with_path(fo, d)
  a <- r$m
  G <- gam(fo, data = d, method = "REML", fit = FALSE)
  X <- as.matrix(d[, cols]); dimnames(X) <- NULL
  out <- c(list(
    x_train = X, y_train = d$y, k = I(k),
    initial_log_sp = I(log(mgcv:::initial.spg(G$X, G$y, G$w, G$family, G$S, G$rank, G$off))),
    strict = list(reml = a$gcv.ubre, sp = I(a$sp), edf_total = sum(a$edf), fitted = a$fitted.values,
                  final_grad_max = max(abs(a$outer.info$grad))),
    mgcv_path = r$path), extra(a))
  write_json(out, sprintf("tests/fixtures/strict/%s.json", name), digits = I(17),
             matrix = "rowmajor", auto_unbox = TRUE)
  cat(name, "REML", format(a$gcv.ubre, digits = 12), "iterations", nrow(r$path), "\n")
}

# --- start basin (seed 45) ---------------------------------------------------
set.seed(45); n <- 300
d <- data.frame(gla=runif(n,1200,3500), lot=rlnorm(n, log(7000), 0.35), garage=sample(0:3,n,TRUE),
                conc=ifelse(runif(n)<0.7,0,runif(n,0,15000)), date=runif(n,0,365), cond=sample(1:6,n,TRUE))
d$y <- 150*d$gla + 3*d$lot + 8000*d$garage - 0.5*d$conc + 40*d$date + 6000*sqrt(d$cond) + rnorm(n, sd=25000)
fo <- y ~ s(gla,k=6,bs="cr")+s(lot,k=6,bs="cr")+s(garage,k=4,bs="cr")+s(conc,k=6,bs="cr")+s(date,k=6,bs="cr")+s(cond,k=6,bs="cr")
write_fixture("gaussian_start_basin_n300", d, c("gla","lot","garage","conc","date","cond"), c(6,6,4,6,6,6), fo,
  function(a) {
    b <- gam(fo, data=d, method="REML", control=ctl, in.out=list(sp=rep(1,6), scale=1))
    list(from_sp1 = list(reml = b$gcv.ubre, sp = I(b$sp), edf_total = sum(b$edf)))
  })

# --- indefinite first steps (seed 180) ---------------------------------------
set.seed(180); n <- 320
d <- data.frame(bath=sample(c(1,1.5,2,2.5,3,3.5),n,TRUE,prob=c(1,1,4,2,2,.5)), bed=sample(2:5,n,TRUE),
  carport=sample(0:2,n,TRUE,prob=c(8,1.5,.5)), conc=ifelse(runif(n)<.7,0,round(runif(n,0,12000),-2)),
  cond=sample(1:6,n,TRUE,prob=c(.2,.5,1,3,2,.5)), date=runif(n,0,540), garage=sample(0:3,n,TRUE,prob=c(.5,1,6,1.5)),
  gla=rlnorm(n,log(1900),.25), lot=rlnorm(n,log(7500),.4), pool=rbinom(n,1,.3), qual=sample(1:6,n,TRUE,prob=c(.2,.5,2,3,1.5,.4)))
d$y <- 180*d$gla + 2*d$lot*runif(1,0,1) + 9000*d$bath + 6000*d$bed + 15000*d$pool - 0.6*d$conc + 30*d$date +
       8000*log(d$cond) + 12000*d$qual + 5000*pmin(d$garage,2) + rnorm(n, sd=runif(1,15000,40000))
cols <- c("bath","bed","carport","conc","cond","date","garage","gla","lot","pool","qual")
fo <- y ~ s(bath,k=4,bs="cr")+bed+s(carport,k=3,bs="cr")+s(conc,k=6,bs="cr")+s(cond,k=6,bs="cr")+s(date,k=6,bs="cr")+s(garage,k=4,bs="cr")+s(gla,k=6,bs="cr")+s(lot,k=6,bs="cr")+pool+s(qual,k=6,bs="cr")
write_fixture("gaussian_indefinite_step_n320", d, cols, c(4,0,3,6,6,6,4,6,6,0,6), fo,
  function(a) {
    # The lower minimum, reached by mgcv from random starts.
    set.seed(1); best <- NULL
    for (i in 1:30) {
      m <- try(gam(fo, data=d, method="REML", control=ctl,
                   in.out=list(sp=exp(runif(length(a$sp), -3, 12)), scale=var(d$y))), silent=TRUE)
      if (!inherits(m, "try-error") && (is.null(best) || m$gcv.ubre < best$gcv.ubre)) best <- m
    }
    stopifnot(best$gcv.ubre < a$gcv.ubre - 0.05)
    list(lower_minimum = list(reml = best$gcv.ubre, sp = I(best$sp)))
  })
