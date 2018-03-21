#' Working with a kde1d object
#'
#' Density, distribution function, quantile function and random generation
#' for a kernel density estimate.
#'
#' @aliases pkde1d, qkde1d, rkde1d
#'
#' @param x vector of density evaluation points.
#' @param obj a \code{kde1d} object.
#'
#' @details \code{\link{dkde1d}} gives the density, \code{\link{pkde1d}} gives
#' the distribution function, \code{\link{qkde1d}} gives the quantile function,
#' and \code{\link{rkde1d}} generates random deviates.
#'
#' The length of the result is determined by `n` for \code{\link{rkde1d}}, and
#' is the length of the numerical argument for the other functions.
#'
#' @return The density, distribution function or quantile functions estimates
#' evaluated respectively at `x`, `q`, or `p`, or a sample of `n` random
#' deviates from the estimated kernel density.
#'
#' @seealso
#' \code{\link{kde1d}}
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

#' @param q vector of quantiles.
#' @rdname dkde1d
#' @export
pkde1d <- function(q, obj) {
    if (is.data.frame(q))
        q <- q[[1]]
    if (!is.ordered(q))
        stopifnot(!is.factor(q))

    q <- expand_as_numeric(q)

    if (length(obj$jitter_info$i_disc) != 1) {
        p <- pkde1d_cpp(q, obj)
    } else {
        # for discrete variables we have to add the missing probability mass
        x_all_num <- expand_as_numeric(as.ordered(obj$jitter_info$levels))
        f_all <- dkde1d(x_all_num, obj)
        p <- sapply(q, function(y) sum(f_all[x_all_num <= y]))
    }

    p
}

#' @param p vector of probabilities.
#' @rdname dkde1d
#' @export
qkde1d <- function(p, obj) {
    stopifnot(all(na.omit(p) > 0.0) & all(na.omit(p) < 1.0))
    if (length(obj$jitter_info$i_disc) != 1) {
        q <- qkde1d_cpp(p, obj)
    } else {
        ## for discrete variables compute quantile from the density
        x_all_num <- expand_as_numeric(as.ordered(obj$jitter_info$levels))

        # pdf at all possible values of x
        dd <- dkde1d(p, obj)
        pp <- c(cumsum(dd)) / sum(dd)

        # generalized inverse
        q <- x_all_num[vapply(p, function(y) which(y <= pp)[1], integer(1))]
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
    structure(object$loglik, "df" = object$edf)
}

#' @method print kde1d
#' @export
print.kde1d <- function(x, ...) {
    if (length(x$jitter_info$i_disc) == 1)
        cat("(jittered) ")
    cat("kernel density estimate ('kde1d')")
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
}

#' @method summary kde1d
#' @export
summary.kde1d <- function(x, ...) {
    print(x)
    cat("---\n")
    cat(paste0("nobs = ", x$nobs, ", "))
    cat(paste0("bw = ", round(x$bw, 2), ", "))
    cat(paste0("loglik = ", round(x$loglik, 2), ", "))
    cat(paste0("d.f. = ", round(x$edf, 2)), "\n")
}
