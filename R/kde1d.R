#' Univariate kernel density estimation for bounded and unbounded support
#'
#' Discrete variables are handled via jittering (see, Nagler,
#' 2017). If a variable should be treated as discrete, declare it as
#' [ordered()].
#'
#' @param x vector of length \eqn{n}.
#' @param mult numeric; the actual bandwidth used is \eqn{bw*mult}.
#' @param xmin lower bound for the support of the density, `NaN` means no
#'   boundary.
#' @param xmax upper bound for the support of the density, `NaN` means no
#'   boundary.
#' @param bw bandwidth parameter; has to be a positive number or \code{NA};
#'   the latter calls [`KernSmooth::dpik()`].
#'
#' @return An object of class \code{kde1d}.
#'
#' @details If \code{xmin} or \code{xmax} are finite, the density estimate will
#'   be 0 outside of \eqn{[xmin, xmax]}. A log-transform is used if there is
#'   only one bounary; a probit transform is used if there are two. Discrete
#'   variables are handled via jittering distribution (see, Nagler, 2017).
#'
#' @seealso \code{\link{dkde1d}}, \code{\link{pkde1d}}, \code{\link{qkde1d}},
#'   \code{\link{rkde1d}} \code{\link{plot.kde1d}} , \code{\link{lines.kde1d}}
#'
#' @references Nagler, T. (2017). *A generic approach to nonparametric function
#'   estimation with mixed data.* [arXiv:1704.07457](https://arxiv.org/abs/1704.07457)
#'
#' @examples
#' data(wdbc, package = "kdecopula")  # load data
#' fit <- kde1d(wdbc[, 5])            # estimate density
#' dkde1d(1000, fit)                  # evaluate density estimate
#'
#' @importFrom KernSmooth dpik
#' @importFrom MASS bandwidth.nrd
#' @importFrom cctools cont_conv
#' @importFrom stats na.omit
#' @export
kde1d <- function(x, mult = 1, xmin = NaN, xmax = NaN, bw = NA) {
    x <- na.omit(x)

    # sanity checks
    check_arguments(x, mult, xmin, xmax, bw)

    # jittering for discrete variables
    x <- cctools::cont_conv(x)

    # bandwidth selection
    bw <- select_bw(boundary_transform(x, xmin, xmax), bw, mult)

    # fit model
    fit <- fit_kde1d_cpp(x, bw, xmin, xmax)

    # return as kde1d object
    fit$jitter_info <- attributes(x)
    class(fit) <- "kde1d"
    fit
}

#' check and pre-process arguments passed to kde1d()
#' @noRd
check_arguments <- function(x, mult, xmin, xmax, bw) {
    stopifnot(NCOL(x) == 1)

    if (!is.ordered(x) & is.factor(x))
        stop("Factors not allowed; use kdevine::kdevine() or cctools::cckde().")

    stopifnot(mult > 0)

    if (is.ordered(x) & (!is.nan(xmin) | !is.nan(xmax)))
        stop("xmin and xmax are not meaningful for x of type ordered.")

    if (!is.nan(xmax) & !is.nan(xmin)) {
        if (xmin > xmax)
            stop("xmin is larger than xmax.")
        if (any(x < xmin) || any(x > xmax))
            stop("Not all data are contained in the interval [xmin, xmax].")
    } else if (!is.nan(xmin)) {
        if (any(x < xmin))
            stop("Not all data are larger than xmin.")
    } else if (!is.nan(xmax)) {
        if (any(x > xmax))
            stop("Not all data are samller than xmax.")
    }
}

#' adjusts observations and evaluation points for boundary effects
#' @noRd
boundary_transform <- function(x, xmin, xmax) {
    if (!is.nan(xmin) & !is.nan(xmax)) {  # two boundaries
        x <- qnorm((x - xmin) / (xmax - xmin + 1e-1))
    } else if (!is.nan(xmin)) {           # left boundary
        x <- log(x - xmin + 1e-3)
    } else if (!is.nan(xmax)) {           # right boundary
        x <- log(xmax - x + 1e-3)
    }

    x
}


#' select's and adjust the bandwidth
#' @noRd
select_bw <- function(x, bw, mult) {
    if (is.na(bw)) {
        # plug in method
        bw <- try(KernSmooth::dpik(x))
        # if it fails: normal rule of thumb
        if (inherits(bw, "try-error"))
            bw <- MASS::bandwidth.nrd(x)
    }

    bw <- mult * bw

    # for discrete use 1 - theta as lower bound for bw
    if (length(attr(x, "i_disc")) == 1) {
        bw <- max(bw, 0.5 - attr(x, "theta"))
    }

    bw
}
