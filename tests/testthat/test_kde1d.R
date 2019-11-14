context("Testing 'kde1d'")

n_sim <- 100
data_types <- c(
  "unbounded", "left_boundary", "right_boundary",
  "two_boundaries", "discrete"
)
deg <- 0:2

scenarios <- expand.grid(data_types = data_types,
                         deg = deg,
                         stringsAsFactors = FALSE)
scenarios <- split(scenarios, seq_len(nrow(scenarios)))
fits <- as.list(seq_along(scenarios))
sims <- as.list(seq_along(scenarios))

for (k in seq_along(scenarios)) {
  test_that(paste0("can fit ", paste(scenarios[[k]], collapse = "/")), {
    xmin <- xmax <- NaN
    if (scenarios[[k]]$data_type == "unbounded") {
      x <- rnorm(n_sim)
    } else if (scenarios[[k]]$data_type == "left_boundary") {
      x <- rexp(n_sim)
      xmin <- 0
    } else if (scenarios[[k]]$data_type == "right_boundary") {
      x <- -rexp(n_sim)
      xmax <- 0
    } else if (scenarios[[k]]$data_type == "two_boundaries") {
      x <- runif(n_sim)
      xmin <- 0
      xmax <- 1
    } else {
      x <- ordered(rbinom(n_sim, size = 5, prob = 0.5), levels = 0:5)
    }
    sims[[k]] <- x
    expect_silent(
      fits[[k]] <<- kde1d(x, xmin = xmin, xmax = xmax, deg = scenarios[[k]]$deg)
    )
  })
}

test_that("detects wrong arguments", {
  x <- rnorm(n_sim)
  expect_error(kde1d(x, xmin = 0))
  expect_error(kde1d(x, xmax = 0))
  expect_error(kde1d(x, xmin = 10, xmax = -10))
  expect_error(kde1d(x, mult = 0))
  expect_error(kde1d(x, bw = -1))
  expect_error(kde1d(x, deg = 3))
  expect_error(supressWarnings(kde1d(x, weights = list())))
  expect_error(kde1d(x, weights = 1:3))

  x <- ordered(rbinom(n_sim, size = 5, prob = 0.5), levels = 0:5)
  expect_error(kde1d(x, xmax = 0))
})

test_that("returns proper 'kde1d' object", {
  lapply(fits, function(x) expect_s3_class(x, "kde1d"))

  class_members <- c(
    "grid_points", "values", "nlevels", "bw", "xmin", "xmax", "deg",
    "edf", "loglik", "x", "weights", "nobs",  "var_name"
  )
  lapply(fits, function(x) expect_identical(names(x), class_members))
})

u <- runif(20)
for (k in seq_along(scenarios)) {
  test_that(paste("d/p/r/h works for", paste(scenarios[[k]], collapse = "/")), {
    fit <- fits[[k]]
    sim <- rkde1d(20, fit)
    if (is.nan(fit$xmax)) {
      xmax <- ifelse(is.ordered(fit$x), 5, Inf)
    } else {
      xmax <- fit$xmax
    }
    if (is.nan(fit$xmin)) {
      xmin <- ifelse(is.ordered(fit$x), 0, -Inf)
    } else {
      xmin <- fit$xmin
    }
    expect_that(all(sim >= xmin), equals(TRUE), label = scenarios)
    expect_that(all(sim <= xmax), equals(TRUE))
    expect_gte(max(dkde1d(sim, fit), 0), 0)
    expect_gte(max(pkde1d(sim, fit), 0), 0)
    expect_lte(max(pkde1d(sim, fit), 1), 1)
    expect_that(all(qkde1d(u, fit) >= xmin), equals(TRUE))
    expect_that(all(qkde1d(u, fit) <= xmax), equals(TRUE))
    if (!is.nan(fit$xmin)) {
      expect_equal(dkde1d(xmin - 1, fit), 0)
      expect_equal(pkde1d(xmin - 1, fit), 0)
    }

    if (!is.nan(fit$xmax)) {
      expect_equal(dkde1d(xmax + 1, fit), 0)
      expect_equal(pkde1d(xmax + 1, fit), 1)
    }

  })
}

test_that("plot functions work", {
  test_plot <- function(fit) {
    expect_silent(plot(fit))
    if (is.ordered(fit$x)) {
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

test_that("behavior for discrete data is consistent", {
  n <- 1e3
  x <- ordered(sample(5, n, TRUE), 1:5)
  fit <- kde1d(x)
  xx <- ordered(1:5, 1:5)
  expect_equal(dkde1d(1:5, fit), dkde1d(xx, fit))
  expect_equal(pkde1d(1:5, fit), pkde1d(xx, fit))
  expect_error(all(is.na(dkde1d(c(0, 6), fit))))
  expect_true(all(rkde1d(n, fit) %in% x))
})

test_that("estimates for discrete data are reasonable", {
  x <- ordered(sample(5, 1e5, TRUE), 1:5)
  fit <- kde1d(x)
  expect_true(all(abs(dkde1d(1:5, fit) - 0.2) < 0.1))
})

test_that("works with weights", {
  n_sim <- 1000
  x <- rnorm(n_sim)

  fit <- kde1d(x, weights = rep(1, n_sim))
  fit0 <- kde1d(x)
  expect_equal(dkde1d(x, fit), dkde1d(x, fit0), tolerance = 0.01)

  fit <- kde1d(x, weights = c(rep(1, n_sim / 2), rep(0, n_sim / 2)))
  fit0 <- kde1d(x[seq_len(n_sim / 2)])
  expect_equal(dkde1d(x, fit), dkde1d(x, fit0), tolerance = 0.01)
})
