context("Testing 'kde1d'")

n_sim <- 100
data_types <- c(
  "unbounded", "left_boundary", "right_boundary",
  "two_boundaries", "discrete_old", "discrete_new", "zero-inflated"
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
    nlevels <- 0
    type <- "continuous"
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
    } else if (scenarios[[k]]$data_type == "discrete_old") {
      x <- ordered(rbinom(n_sim, size = 5, prob = 0.5), levels = 0:5)
    } else if (scenarios[[k]]$data_type == "discrete_new") {
      x <- rbinom(n_sim, size = 5, prob = 0.5)
      type <- "discrete"
    } else if (scenarios[[k]]$data_type == "zero-inflated") {
      x <- rexp(n_sim)
      x[sample(1:n_sim, floor(n_sim / 3))] <- 0
      type <- "zi"
    }

    sims[[k]] <- x
    expect_silent(
      fits[[k]] <<- kde1d(x, xmin = xmin, xmax = xmax, type = type,
                          deg = scenarios[[k]]$deg)
    )
  })
}

test_that("detects wrong arguments", {
  x <- rnorm(n_sim)
  expect_error(kde1d(x, xmin = mean(x)))
  expect_error(kde1d(x, xmax = mean(x)))
  expect_error(kde1d(x, xmin = 10, xmax = -10))
  expect_error(kde1d(x, mult = 0))
  expect_error(kde1d(x, bw = -1))
  expect_error(kde1d(x, deg = 3))
  expect_error(supressWarnings(kde1d(x, weights = list())))
  expect_error(kde1d(x, weights = 1:3))
})

test_that("returns proper 'kde1d' object", {
  lapply(fits, function(x) expect_s3_class(x, "kde1d"))

  class_members <- c(
    "grid_points", "values", "xmin", "xmax", "type", "bw", "mult", "deg",
    "boundary_repair", "prob0", "edf", "loglik", "x", "weights", "nobs",
    "var_name"
  )
  lapply(fits, function(x) expect_identical(names(x), class_members))
})

test_that("boundary repair can be disabled", {
  probabilities <- seq(0.5 / 200, 1 - 0.5 / 200, length.out = 200)
  observations <- qexp(probabilities)
  repaired <- kde1d(observations, xmin = 0)
  bulk <- kde1d(observations, xmin = 0, boundary_repair = FALSE)

  expect_true(repaired$boundary_repair)
  expect_false(bulk$boundary_repair)
  expect_false(isTRUE(all.equal(repaired$values, bulk$values)))
})

test_that("finite bounds support discrete and zero-inflated data", {
  discrete <- kde1d(
    rep(-2:1, 30), xmin = -2, xmax = 1, type = "discrete"
  )
  expect_equal(sum(dkde1d(-2:1, discrete)), 1)
  expect_equal(dkde1d(c(-3, 2), discrete), c(0, 0))
  expect_error(kde1d(c(0, 1.5), type = "discrete"))
  expect_error(kde1d(0:2, xmin = 0.5, type = "discrete"))

  observations <- c(rep(0, 40), seq(1.01, 1.99, length.out = 160))
  zero_inflated <- kde1d(
    observations, xmin = 1, xmax = 2, type = "zero_inflated"
  )
  expect_equal(dkde1d(0, zero_inflated), 0.2)
  expect_equal(dkde1d(0.5, zero_inflated), 0)
  expect_error(kde1d(c(0, 0.5, 1.5), xmin = 1, xmax = 2, type = "zi"))
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
    sim[c(2, 5, 9)] <- NA
    expect_gte(max(na.omit(dkde1d(sim, fit)), 0), 0)
    expect_gte(max(na.omit(pkde1d(sim, fit)), 0), 0)
    expect_lte(max(na.omit(pkde1d(sim, fit)), 1), 1)
    expect_that(all(na.omit(qkde1d(u, fit) >= xmin)), equals(TRUE))
    expect_that(all(na.omit(qkde1d(u, fit) <= xmax)), equals(TRUE))
    if (!(fit$type == "discrete") & !is.nan(fit$xmin)) {
      expect_equal(dkde1d(xmin - 1, fit), 0)
      expect_equal(pkde1d(xmin - 1, fit), 0)
    }

    if (!(fit$type == "discrete") & !is.nan(fit$xmax)) {
      expect_equal(dkde1d(xmax + 1, fit), 0)
      expect_equal(pkde1d(xmax + 1, fit), 1)
    }
  })
}

test_that("plot functions work", {
  test_plot <- function(fit) {
    expect_silent(plot(fit))
    expect_silent(lines(fit))
    expect_silent(points(fit))
  }

  lapply(fits, test_plot)
})

test_that("other generics work", {
  test_other_generics <- function(fit) {
    expect_output(print(fit))
    expect_output(s <- summary(fit))
    expect_is(s, "numeric")
    expect_equal(length(s), 5)
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

test_that("reports the weighted log-likelihood", {
  set.seed(1)
  observations <- rnorm(100)
  weights <- rexp(100)
  fit <- kde1d(observations, weights = weights)

  expect_equal(
    fit$loglik,
    sum(weights / mean(weights) * log(dkde1d(observations, fit)))
  )
})

test_that("includes the point mass in zero-inflated log-likelihood", {
  set.seed(2)
  observations <- c(rep(0, 40), rexp(60))
  fit <- kde1d(observations, xmin = 0, type = "zero-inflated")

  expect_equal(fit$loglik, sum(log(dkde1d(observations, fit))))
})

test_that("one-sided bounded estimates are scale equivariant", {
  set.seed(3)
  observations <- rexp(300)
  scale <- 1e6
  probabilities <- seq(0.05, 0.95, length.out = 31)
  evaluation_points <- quantile(observations, probabilities, names = FALSE)

  check_equivariance <- function(x, boundary, eval) {
    if (boundary == "left") {
      fit <- kde1d(x, xmin = 0)
      fit_scaled <- kde1d(x * scale, xmin = 0)
    } else {
      fit <- kde1d(x, xmax = 0)
      fit_scaled <- kde1d(x * scale, xmax = 0)
    }

    expect_equal(
      dkde1d(eval, fit),
      scale * dkde1d(eval * scale, fit_scaled),
      tolerance = 1e-10
    )
    expect_equal(
      pkde1d(eval, fit),
      pkde1d(eval * scale, fit_scaled),
      tolerance = 1e-10
    )
    expect_equal(
      qkde1d(probabilities, fit),
      qkde1d(probabilities, fit_scaled) / scale,
      tolerance = 1e-10
    )
  }

  check_equivariance(observations, "left", evaluation_points)
  check_equivariance(-observations, "right", -evaluation_points)
})

test_that("quantiles work for fully zero-inflated estimates", {
  fit <- kde1d(rep(0, 20), xmin = 0, type = "zero-inflated")

  expect_equal(qkde1d(c(0, 0.25, 0.5, 0.75, 1), fit), rep(0, 5))
  expect_true(is.nan(qkde1d(NaN, fit)))
})
