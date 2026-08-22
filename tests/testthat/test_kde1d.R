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
    "prob0", "edf", "loglik", "x", "weights", "nobs",  "var_name"
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

test_that("bounded fits are invariant under reflection", {
  set.seed(3)
  observations <- runif(500)
  fit <- kde1d(observations, xmin = 0, xmax = 1)
  reflected_fit <- kde1d(1 - observations, xmin = 0, xmax = 1)

  expect_equal(fit$bw, reflected_fit$bw)
  expect_equal(fit$grid_points, 1 - rev(reflected_fit$grid_points))
  expect_equal(fit$values, rev(reflected_fit$values))
})

test_that("one-boundary fits are invariant under reflection", {
  set.seed(7)
  observations <- rexp(500)
  fit <- kde1d(observations, xmin = 0)
  reflected_fit <- kde1d(-observations, xmax = 0)

  expect_equal(fit$bw, reflected_fit$bw)
  expect_equal(fit$grid_points, -rev(reflected_fit$grid_points))
  expect_equal(fit$values, rev(reflected_fit$values))
  expect_equal(fit$loglik, reflected_fit$loglik)
  expect_equal(fit$edf, reflected_fit$edf)
})

test_that("boundary grids resolve the support beyond the observations", {
  set.seed(5)

  observations <- runif(500, 0.2, 0.8)
  fit <- kde1d(observations, xmin = 0, xmax = 1, bw = 0.3)
  expect_true(all(diff(fit$grid_points) > 0))
  expect_equal(range(fit$grid_points), c(0, 1))
  expect_lt(fit$grid_points[2], min(observations))
  expect_gt(fit$grid_points[length(fit$grid_points) - 1], max(observations))

  observations <- rexp(500)
  fit <- kde1d(observations, xmin = 0, bw = 0.3)
  expect_true(all(diff(fit$grid_points) > 0))
  expect_equal(head(fit$grid_points, 1), 0)
  expect_lt(fit$grid_points[2], min(observations))
  expect_gt(tail(fit$grid_points, 1), max(observations))

  observations <- -rexp(500)
  fit <- kde1d(observations, xmax = 0, bw = 0.3)
  expect_true(all(diff(fit$grid_points) > 0))
  expect_equal(tail(fit$grid_points, 1), 0)
  expect_lt(head(fit$grid_points, 1), min(observations))
  expect_gt(fit$grid_points[length(fit$grid_points) - 1], max(observations))
})

test_that("two-boundary fits are affine equivariant", {
  set.seed(6)
  observations <- rbeta(500, 2, 3)
  fit <- kde1d(observations, xmin = 0, xmax = 1)
  scaled_fit <- kde1d(2 + 3 * observations, xmin = 2, xmax = 5)

  expect_equal(scaled_fit$bw, fit$bw)
  expect_equal(scaled_fit$grid_points, 2 + 3 * fit$grid_points)
  expect_equal(scaled_fit$values, fit$values / 3)
})

test_that("density interpolation is continuous at the right grid endpoint", {
  set.seed(4)
  fit <- kde1d(rnorm(500))

  expect_equal(
    dkde1d(tail(fit$grid_points, 1), fit),
    tail(fit$values, 1)
  )
})
