#' Working with a kde1d object
#'
#' The density, cdf, or quantile function of a kernel density estimate are
#' evaluated at arbitrary points with \code{\link{dkde1d}}, \code{\link{pkde1d}},
#' and \code{\link{qkde1d}} respectively.
#'
#' @aliases pkde1d, qkde1d, rkde1d
#'
#' @param x vector of evaluation points.
#' @param obj a \code{kde1d} object.
#'
#'
#' @return The density or cdf estimate evaluated at \code{x}.
#'
#' @seealso
#' \code{\link{kde1d}}
#'
#' @examples
#' data(wdbc)  # load data
#' fit <- kde1d(wdbc[, 5])  # estimate density
#' dkde1d(1000, fit)        # evaluate density estimate
#' pkde1d(1000, fit)        # evaluate corresponding cdf
#' qkde1d(0.5, fit)         # quantile function
#' hist(rkde1d(100, fit))   # simulate
#'
#' @importFrom cctools expand_as_numeric
#' @export
dkde1d <- function(x, obj) {
    if (is.data.frame(x))
        x <- x[[1]]
    if (!is.ordered(x))
        stopifnot(!is.factor(x))

    x <- expand_as_numeric(x)
    fhat <- dkde1d_cpp(x, obj)

    if (length(obj$jitter_info$i_disc) == 1) {
        # for discrete variables we can normalize
        x_all_num <- expand_as_numeric(as.ordered(obj$jitter_info$levels))
        f_all <- dkde1d_cpp(x_all_num, obj)
        fhat <- fhat / sum(f_all)
    }

    as.vector(fhat)
}


#' @rdname dkde1d
#' @export
pkde1d <- function(x, obj) {
    if (is.data.frame(x))
        x <- x[[1]]
    if (!is.ordered(x))
        stopifnot(!is.factor(x))

    x <- expand_as_numeric(x)

    if (length(obj$jitter_info$i_disc) != 1) {
        p <- pkde1d_cpp(x, obj)
    } else {
        # for discrete variables we have to add the missing probability mass
        x_all_num <- expand_as_numeric(as.ordered(obj$jitter_info$levels))
        f_all <- dkde1d(x_all_num, obj)
        p <- sapply(x, function(y) sum(f_all[x_all_num <= y]))
    }

    p
}

#' @rdname dkde1d
#' @export
qkde1d <- function(x, obj) {
    stopifnot(all(na.omit(x) > 0.0) & all(na.omit(x) < 1.0))
    if (length(obj$jitter_info$i_disc) != 1) {
        q <- qkde1d_cpp(x, obj)
    } else {
        ## for discrete variables compute quantile from the density
        x_all_num <- expand_as_numeric(as.ordered(obj$jitter_info$levels))

        # pdf at all possible values of x
        dd <- dkde1d(x, obj)
        pp <- c(cumsum(dd)) / sum(dd)

        # generalized inverse
        q <- x_all_num[vapply(x, function(y) which(y <= pp)[1], integer(1))]
        q <- ordered(obj$fitter_info$levels[q], levels = obj$fitter_info$levels)
    }

    q
}

#' @param n integer; number of observations.
#' @param quasi logical; the default (\code{FALSE}) returns pseudo-random
#' numbers, use \code{TRUE} for quasi-random numbers (generalized Halton, see
#' \code{\link[qrng:ghalton]{ghalton}}).
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
#' @param x \code{kde1d} object.
#' @param ... further arguments passed to \code{\link{plot.default}}.
#'
#' @seealso
#' \code{\link{kde1d}}
#' \code{\link{lines.kde1d}}
#'
#' @examples
#' data(wdbc)  # load data
#' fit <- kde1d(wdbc[, 7])  # estimate density
#' plot(fit)  # plot density estimate
#'
#' fit2 <- kde1d(as.ordered(wdbc[, 1])) # discrete variable
#' plot(fit2, col = 2)
#'
#' @importFrom graphics plot
#' @importFrom utils modifyList
#' @export
plot.kde1d <- function(x, ...) {
    plot_type <- "l"  # for continuous variables, use a line plot
    if (length(x$jitter_info$i_disc) == 1) {
        ev <- as.ordered(x$jitter_info$levels)
        plot_type <- "h"  # for discrete variables, use a histrogram
    }

    pars <- list(
        x = x$grid_points,
        y = x$values,
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

#' @method logLik kde1d
#'
#' @export
logLik.kde1d <- function(object, ...) {
    structure(object$fit$loglik, "df" = object$fit$edf)
}

