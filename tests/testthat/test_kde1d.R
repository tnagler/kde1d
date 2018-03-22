context("Testing 'kde1d'")

set.seed(0)
n_sim <- 1e2
data_types <- c("unbounded", "left_boundary", "right_boundary",
                "two_boundaries", "discrete")
deg <- 0:2

scenarios <- expand.grid(data_types = data_types, deg = deg)
scenarios <- split(scenarios, seq_len(nrow(scenarios)))
fits <- lapply(scenarios,
               function(scenario) {
                   xmin <- xmax <- NaN
                   if (scenario$data_type == "unbounded") {
                       x <- rnorm(n_sim)
                   } else if (scenario$data_type == "left_boundary") {
                       x <- rexp(n_sim)
                       xmin <- 0
                   }  else if (scenario$data_type == "right_boundary") {
                       x <- -rexp(n_sim)
                       xmax <- 0
                   }  else if (scenario$data_type == "two_boundaries") {
                       x <- runif(n_sim)
                       xmin <- 0
                       xmax <- 1
                   } else {
                       x <- ordered(rbinom(n_sim, size = 5, prob = 0.5),
                                    levels = 0:5)
                   }
                   kde1d(x, xmin = xmin, xmax = xmax, deg = scenario$deg)
               })

test_that("detects wrong arguments", {
    x <- rnorm(n_sim)
    expect_error(kde1d(x, xmin = 0))
    expect_error(kde1d(x, xmax = 0))
    expect_error(kde1d(x, xmin = 10, xmax = -10))
    expect_error(kde1d(x, mult = 0))
    expect_error(kde1d(x, deg = 3))

    x <- ordered(rbinom(n_sim, size = 5, prob = 0.5), levels = 0:5)
    expect_error(kde1d(x, xmax = 0))
})

test_that("returns proper 'kde1d' object", {
    lapply(fits, function(x) expect_s3_class(x, "kde1d"))

    class_members <- c("grid_points", "values", "bw", "xmin", "xmax", "deg",
                       "edf", "loglik", "jitter_info", "var_name", "nobs")
    lapply(fits, function(x) expect_identical(names(x), class_members))
})

test_that("d/p/r/h functions work", {
    n <- 50
    u <- runif(n)
    test_dpqr <- function(fit, sim) {
        sim <- data.frame(sim)
        is_jittered <- length(fit$jitter_info$i_disc) == 1
        if (is.nan(fit$xmax)) {
            xmax <- ifelse(is_jittered, fit$jitter_info$nu, Inf)
        } else {
            xmax <- fit$xmax
        }
        if (is.nan(fit$xmin)) {
            xmin <- ifelse(is_jittered, 0, -Inf)
        } else {
            xmin <- fit$xmin
        }
        expect_that(all(sim >= xmin), equals(TRUE))
        expect_that(all(sim <= xmax), equals(TRUE))
        expect_gte(max(dkde1d(sim, fit), 0), 0)
        expect_gte(max(pkde1d(sim, fit), 0), 0)
        expect_lte(max(pkde1d(sim, fit), 1), 1)
        expect_that(all(qkde1d(u, fit) >= xmin), equals(TRUE))
        expect_that(all(qkde1d(u, fit) <= xmax), equals(TRUE))
    }

    sims <- lapply(fits, function(x) rkde1d(n, x))
    mapply(test_dpqr, fits, sims)

    sim <- lapply(fits, function(x) rkde1d(n, x, quasi = TRUE))
    mapply(test_dpqr, fits, sims)
})

test_that("plot functions work", {

    test_plot <- function(fit) {
        expect_silent(plot(fit))
        if (length(fit$jitter_info$i_disc) == 1) {
            expect_error(lines(fit))
        } else {
            expect_silent(lines(fit))
        }
    }

    lapply(fits, test_plot)
})

test_that("other generics work", {

    test_other_generics <- function(fit) {
        expect_output(print(fit))
        expect_output(s <- summary(fit))
        expect_is(s, "numeric")
        expect_equal(length(s), 4)
        expect_silent(s <- logLik(fit))
        expect_is(s, "numeric")
    }

    lapply(fits, test_other_generics)
})