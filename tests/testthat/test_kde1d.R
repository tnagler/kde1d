context("Testing 'kde1d'")

set.seed(0)
fits <- lapply(c("unbounded", "bounded", "discrete"),
               function(type) {
                   xmin <- xmax <- NaN
                   if (type == "unbounded") {
                       x <- rnorm(1e2)
                   } else if (type == "bounded") {
                       x <- runif(1e2)
                       xmin <- 0
                       xmax <- 1
                   } else {
                       x <- ordered(rbinom(1e2, size = 5, prob = 0.5),
                                    levels = 0:5)
                   }
                   return(kde1d(x, xmin = xmin, xmax = xmax))
               })

test_that("returns proper 'kde1d' object", {
    lapply(fits, function(x) expect_s3_class(x, "kde1d"))

    class_members <- c("grid_points", "values", "bw", "xmin", "xmax", "edf",
                       "loglik", "jitter_info", "var_name", "nobs")
    lapply(fits, function(x) expect_identical(names(x), class_members))
})

test_that("d/p/r/h functions work", {
    n <- 50
    u <- runif(n)
    test_dpqr <- function(fit, sim) {
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

test_that("print/summary generics work", {

    test_print_summary <- function(fit) {
        expect_output(print(fit))
        expect_output(s <- summary(fit))
        expect_is(s, "numeric")
        expect_equal(length(s), 4)
    }

    lapply(fits, test_print_summary)
})