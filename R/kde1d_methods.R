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
#' set.seed(0)              # for reproducibility
#' x <- rnorm(100)          # simulate some data
#' fit <- kde1d(x)          # estimate density
#' dkde1d(0, fit)           # evaluate density estimate (close to dnorm(0))
#' pkde1d(0, fit)           # evaluate corresponding cdf (close to pnorm(0))
#' qkde1d(0.5, fit)         # quantile function (close to qnorm(0))
#' hist(rkde1d(100, fit))   # simulate
#'
#' @importFrom cctools expand_as_numeric
#' @export
dkde1d <- function(x, obj) {
    if (is.data.frame(x))
        x <- x[[1]]
    if (!is.ordered(x))
        stopifnot(!is.factor(x))

    # adjust grid to stabilize estimates
    rng <- diff(range(obj$grid_points))
    if (!is.nan(obj$xmin))
        obj$grid_points[1] <- obj$xmin - 0.1 * rng
    if (!is.nan(obj$xmax))
        obj$grid_points[length(obj$grid_points)] <- obj$xmax + 0.1 * rng

    if (length(obj$jitter_info$i_disc) == 1 & !is.ordered(x))
        x <- ordered(x, obj$jitter_info$levels$x)

    fhat <- dkde1d_cpp(expand_as_numeric(x), obj)

    if (length(obj$jitter_info$i_disc) == 1) {
        # for discrete variables we can normalize
        f_all <- dkde1d_cpp(seq_along(obj$jitter_info$levels$x) - 1, obj)
        fhat <- fhat / sum(f_all)
    }

    as.vector(fhat)
}

#' @param q vector of quantiles.
#' @rdname dkde1d
#' @export
pkde1d <- function(q, obj) {
    if (is.data.frame(q))
        q <- q[[1]]
    if (!is.ordered(q))
        stopifnot(!is.factor(q))

    if (length(obj$jitter_info$i_disc) != 1) {
        p <- pkde1d_cpp(q, obj)
    } else {
        if (!is.ordered(q))
            q <- ordered(q, obj$jitter_info$levels$x)
        x_all <- as.ordered(obj$jitter_info$levels$x)
        p_all <- dkde1d(x_all, obj)
        p <- sapply(q, function(y) sum(p_all[x_all <= y]))
        p <- pmin(pmax(p, 0), 1)
    }

    p
}

#' @param p vector of probabilities.
#' @rdname dkde1d
#' @export
qkde1d <- function(p, obj) {
    stopifnot(all(na.omit(p) > 0.0) & all(na.omit(p) < 1.0))
    if (length(obj$jitter_info$i_disc) != 1) {
        ## qkde1d_cpp for continuous variables
        q <- qkde1d_cpp(p, obj)
    } else {
        ## for discrete variables compute quantile from the density
        x_all <- as.ordered(obj$jitter_info$levels$x)

        # pdf at all possible values of x
        pp <- pkde1d(x_all, obj)

        # generalized inverse
        q <- x_all[vapply(p, function(y) which(y <= pp)[1], integer(1))]
    }

    q
}

#' @param n integer; number of observations.
#' @param quasi logical; the default (`FALSE`) returns pseudo-random
#' numbers, use `TRUE` for quasi-random numbers (generalized Halton, see
#' [`qrng::ghalton()`]).
#'
#' @rdname dkde1d
#' @importFrom qrng ghalton
#' @importFrom stats runif
#' @export
rkde1d <- function(n, obj, quasi = FALSE) {
    # simulate (psuedo/quasi) uniform random variables
    if (!quasi) {
        U <- runif(n)
    } else {
        U <- ghalton(n, d = 1)
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
#' x <- rbeta(100, shape1 = 0.3, shape2 = 0.4)   # simulate data
#' fit <- kde1d(x)                               # unbounded estimate
#' plot(fit, ylim = c(0, 4))                     # plot estimate
#' curve(dbeta(x, 0.3, 0.4),                     # add true density
#'       col = "red", add = TRUE)
#' fit_bounded <- kde1d(x, xmin = 0, xmax = 1)   # bounded estimate
#' lines(fit_bounded, col = "green")
#'
#' ## discrete data
#' x <- rpois(100, 3)                        # simulate data
#' x <- ordered(x, levels = 0:20)            # declare variable as ordered
#' fit <- kde1d(x)                           # estimate density
#' plot(fit, ylim = c(0, 0.25))              # plot density estimate
#' points(ordered(0:20, 0:20),               # add true density values
#'        dpois(0:20, 3), col = "red")
#'
#' @importFrom graphics plot
#' @importFrom utils modifyList
#' @export
plot.kde1d <- function(x, ...) {
    plot_type <- "l"  # for continuous variables, use a line plot
    if (length(x$jitter_info$i_disc) == 1) {
        ev <- ordered(x$jitter_info$levels$x,
                      levels = x$jitter_info$levels$x)
        plot_type <- "h"  # for discrete variables, use a histrogram
    } else {
        # adjust grid if necessary
        ev <- x$grid_points
        if (!is.nan(x$xmin))
            ev[1] <- x$xmin
        if (!is.nan(x$xmax))
            ev[length(ev)] <- x$xmax
    }
    vals <- dkde1d(ev, x)

    pars <- list(
        x = ev,
        y = vals,
        type = plot_type,
        xlab = "x",
        ylab = "density",
        ylim = c(0, 1.1 * max(x$values))
    )

    do.call(plot, modifyList(pars, list(...)))
}

#' @method lines kde1d
#'
#' @rdname plot.kde1d
#' @importFrom graphics lines
#' @importFrom utils modifyList
#' @export
lines.kde1d <- function(x, ...) {
    if (length(x$jitter_info$i_disc) == 1)
        stop("lines does not work for discrete estimates.")
    pars <- list(x = x$grid_points, y = x$values)
    do.call(lines, modifyList(pars, list(...)))
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
    if (length(x$jitter_info$i_disc) == 1)
        cat("(jittered) ")
    cat("kernel density estimate ('kde1d')")
    if (x$deg > 0) {
        if (x$deg == 1)
            cat(", log-linear")
        if (x$deg == 2)
            cat(", log-quadratic")
    }
    if (!is.nan(x$xmin) | !is.nan(x$xmax)) {
        cat(" with bounded support (")
        if (!is.nan(x$xmin))
            cat("xmin =", round(x$xmin, 2))
        if (!is.nan(x$xmax)) {
            if (!is.nan(x$xmin))
                cat(", ")
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

    df <- rep(NA, 4)
    names(df) <- c("nobs", "bw", "loglik", "d.f.")
    df[1] <- object$nobs
    df[2] <- object$bw
    df[3] <- object$loglik
    df[4] <- object$edf

    print(object)
    cat(strrep("-", 65), "\n", sep = "")
    cat(paste(names(df), round(df, 2), sep = " = ", collapse = ", "))
    cat("\n")
    invisible(df)
}
