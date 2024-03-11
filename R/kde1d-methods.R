#' Working with a kde1d object
#'
#' Density, distribution function, quantile function and random generation
#' for a 'kde1d' kernel density estimate.
#'
#' @aliases pkde1d, qkde1d, rkde1d
#'
#' @param x vector of density evaluation points.
#' @param obj a `kde1d` object.
#'
#' @details [`dkde1d()`] gives the density, [`pkde1d()`] gives
#' the distribution function, [`qkde1d()`] gives the quantile function,
#' and [`rkde1d()`] generates random deviates.
#'
#' The length of the result is determined by `n` for [`rkde1d()`], and
#' is the length of the numerical argument for the other functions.
#'
#' @return The density, distribution function or quantile functions estimates
#' evaluated respectively at `x`, `q`, or `p`, or a sample of `n` random
#' deviates from the estimated kernel density.
#'
#' @seealso [`kde1d()`]
#'
#' @examples
#' set.seed(0) # for reproducibility
#' x <- rnorm(100) # simulate some data
#' fit <- kde1d(x) # estimate density
#' dkde1d(0, fit) # evaluate density estimate (close to dnorm(0))
#' pkde1d(0, fit) # evaluate corresponding cdf (close to pnorm(0))
#' qkde1d(0.5, fit) # quantile function (close to qnorm(0))
#' hist(rkde1d(100, fit)) # simulate
#' @export
dkde1d <- function(x, obj) {
  x <- prep_eval_arg(x, obj)
  dkde1d_cpp(x, obj)
}

#' @param q vector of quantiles.
#' @rdname dkde1d
#' @export
pkde1d <- function(q, obj) {
  q <- prep_eval_arg(q, obj)
  pkde1d_cpp(q, obj)
}

#' @param p vector of probabilities.
#' @rdname dkde1d
#' @export
qkde1d <- function(p, obj) {
  q <- qkde1d_cpp(p, obj)
  if (is.ordered(obj$x)) {
    ## for discrete variables, add factor levels
    q <- ordered(levels(obj$x)[q + 1], levels(obj$x))
  }
  q
}

#' @param n integer; number of observations.
#' @param quasi logical; the default (`FALSE`) returns pseudo-random
#' numbers, use `TRUE` for quasi-random numbers (generalized Halton, see
#' [`randtoolbox::sobol()`]).
#'
#' @rdname dkde1d
#' @importFrom randtoolbox sobol
#' @importFrom stats runif
#' @export
rkde1d <- function(n, obj, quasi = FALSE) {
  # simulate (psuedo/quasi) uniform random variables
  if (!quasi) {
    U <- runif(n)
  } else {
    U <- randtoolbox::sobol(n, scrambling = 1, seed = sample.int(1e10, 1))
  }
  # simulated data from KDE is the quantile transform of U
  qkde1d(U, obj)
}

#' Plotting kde1d objects
#'
#' @aliases lines.kde1d
#' @method plot kde1d
#'
#' @param x `kde1d` object.
#' @param ... further arguments passed to [`plot.default()`]
#'
#' @seealso [`kde1d()`]
#'
#' @examples
#' ## continuous data
#' x <- rbeta(100, shape1 = 0.3, shape2 = 0.4) # simulate data
#' fit <- kde1d(x) # unbounded estimate
#' plot(fit, ylim = c(0, 4)) # plot estimate
#' curve(dbeta(x, 0.3, 0.4), # add true density
#'   col = "red", add = TRUE
#' )
#' fit_bounded <- kde1d(x, xmin = 0, xmax = 1) # bounded estimate
#' lines(fit_bounded, col = "green")
#'
#' ## discrete data
#' x <- rpois(100, 3) # simulate data
#' x <- ordered(x, levels = 0:20) # declare variable as ordered
#' fit <- kde1d(x) # estimate density
#' plot(fit, ylim = c(0, 0.25)) # plot density estimate
#' points(ordered(0:20, 0:20), # add true density values
#'   dpois(0:20, 3),
#'   col = "red"
#' )
#'
#' ## zero-inflated data
#' x <- rexp(500, 0.5)  # simulate data
#' x[sample(1:500, 200)] <- 0 # add zero-inflation
#' fit <- kde1d(x, xmin = 0, type = "zi") # estimate density
#' plot(fit) # plot the density estimate
#' lines(  # add true density
#'   seq(0, 20, l = 100),
#'   0.6 * dexp(seq(0, 20, l = 100), 0.5),
#'   col = "red"
#' )
#' points(0, 0.4, col = "red")
#'
#' @importFrom graphics plot
#' @importFrom utils modifyList
#' @export
plot.kde1d <- function(x, ...) {
  ev <- make_plotting_grid(x)
  vals <- dkde1d(ev, x)
  plot_type <- ifelse(x$type == "discrete", "p", "l")
  pars <- list(
    x = ev,
    y = vals,
    type = plot_type,
    xlab = "x",
    ylab = "density",
    ylim = c(0, 1.1 * max(x$values))
  )
  do.call(plot, modifyList(pars, list(...)))

  if (x$type == "zero-inflated") {
    points(0, dkde1d(0, x))
  }
}

#' @method lines kde1d
#'
#' @rdname plot.kde1d
#' @importFrom graphics lines
#' @importFrom utils modifyList
#' @export
lines.kde1d <- function(x, ...) {
  if (x$type == "discrete") {
    points(x, ...)
  }
  ev <- make_plotting_grid(x)
  vals <- dkde1d(ev, x)
  pars <- list(x = ev, y = vals)
  do.call(lines, modifyList(pars, list(...)))

  if (x$type == "zero-inflated") {
    points(0, dkde1d(0, x))
  }
}

#' @method points kde1d
#'
#' @rdname plot.kde1d
#' @importFrom graphics points
#' @importFrom utils modifyList
#' @export
points.kde1d <- function(x, ...) {
  ev <- make_plotting_grid(x)
  vals <- dkde1d(ev, x)
  pars <- list(x = ev, y = vals)
  do.call(points, modifyList(pars, list(...)))
}

make_plotting_grid <- function(x) {
  if (is.ordered(x$x)) {
    ev <- ordered(levels(x$x), levels(x$x))
  } else if (x$type == "discrete") {
    ev <- seq.int(floor(min(x$grid_points)), ceiling(max(x$grid_points)))
  } else {
    # adjust grid if necessary
    ev <- seq(min(x$grid_points), max(x$grid_points), l = 200)
    if (!is.nan(x$xmin)) {
      ev[1] <- x$xmin
    }
    if (!is.nan(x$xmax)) {
      ev[length(ev)] <- x$xmax
    }
    if (x$type == "zero-inflated") {
      ev <- setdiff(ev, 0)
    }
  }
  ev
}

#' @importFrom stats logLik
#' @method logLik kde1d
#'
#' @export
logLik.kde1d <- function(object, ...) {
  structure(object$loglik, "df" = object$edf)
}

#' @method print kde1d
#' @export
print.kde1d <- function(x, ...) {
  if (x$type == "discrete") {
    cat("(jittered) ")
  } else  if (x$type == "zero-inflated") {
    cat("(zero-inflated) ")
  }
  cat("kernel density estimate ('kde1d')")
  if (x$deg > 0) {
    if (x$deg == 1) {
      cat(", log-linear")
    }
    if (x$deg == 2) {
      cat(", log-quadratic")
    }
  }
  if (!is.nan(x$xmin) | !is.nan(x$xmax)) {
    cat(" with bounded support (")
    if (!is.nan(x$xmin)) {
      cat("xmin =", round(x$xmin, 2))
    }
    if (!is.nan(x$xmax)) {
      if (!is.nan(x$xmin)) {
        cat(", ")
      }
      cat("xmax =", round(x$xmax, 2))
    }
    cat(")")
  }
  cat("\n")
  invisible(x)
}

#' @method summary kde1d
#' @export
summary.kde1d <- function(object, ...) {
  df <- rep(NA, 5)
  names(df) <- c("nobs", "bw", "mult", "loglik", "d.f.")
  df[1] <- object$nobs
  df[2] <- object$bw
  df[3] <- object$mult
  df[4] <- object$loglik
  df[5] <- object$edf

  print(object)
  cat(strrep("-", 65), "\n", sep = "")
  cat(paste(names(df), round(df, 2), sep = " = ", collapse = ", "))
  cat("\n")
  invisible(df)
}
