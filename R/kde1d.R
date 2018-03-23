#' Univariate local-polynomial likelihood kernel density estimation
#'
#' The estimator can handle for bounded, unbounded, and discrete support, see
#' *Details*.
#'
#' @param x vector (or one-column matrix/data frame) of observations; can be
#'   `numeric` or `ordered`.
#' @param xmin lower bound for the support of the density (only for continuous
#'   data); `NaN` means no boundary.
#' @param xmax upper bound for the support of the density (only for continuous
#'   data); `NaN` means no boundary.
#' @param mult positive bandwidth multiplier; the actual bandwidth used is
#'   \eqn{bw*mult}.
#' @param bw bandwidth parameter; has to be a positive number or `NA`; the
#'   latter uses the direct plug-in methodology of Sheather and Jones (1991).
#' @param deg degree of the polynomial; either `0`, `1`, or `2` for log-constant,
#'   log-linear, and log-quadratic fitting, respectively.
#'
#' @return An object of class `kde1d`.
#'
#' @details A gaussian kernel is used in all cases. If `xmin` or `xmax` are
#'   finite, the density estimate will be 0 outside of \eqn{[xmin, xmax]}. A
#'   log-transform is used if there is only one boundary (see, Geenens and Wang,
#'   2018); a probit transform is used if there are two (see, Geenens, 2014).
#'   Discrete variables are handled via jittering (see, Nagler, 2018a, 2018b).
#'
#' @seealso [`dkde1d()`], [`pkde1d()`], [`qkde1d()`], [`rkde1d()`],
#'   [`plot.kde1d()`], [`lines.kde1d()`]
#'
#' @references
#'   Geenens, G. (2014). *Probit transformation for kernel density estimation on
#'   the unit interval*. Journal of the American Statistical Association,
#'   109:505, 346-358, [arXiv:1303.4121](https://arxiv.org/abs/1303.4121)
#'
#'   Geenens, G., Wang, C. (2018). *Local-likelihood transformation kernel
#'   density estimation for positive random variables.* Journal of Computational
#'   and Graphical Statistics, to appear,
#'   [arXiv:1602.04862](https://arxiv.org/abs/1602.04862)
#'
#'   Nagler, T. (2018a). *A generic approach to nonparametric function
#'   estimation with mixed data.* Statistics & Probability Letters, 137:326–330,
#'   [arXiv:1704.07457](https://arxiv.org/abs/1704.07457)
#'
#'   Nagler, T. (2018b). *Asymptotic analysis of the jittering kernel density
#'   estimator.* Mathematical Methods of Statistics, in press,
#'   [arXiv:1705.05431](https://arxiv.org/abs/1705.05431)
#'
#'   Sheather, S. J. and Jones, M. C. (1991). A reliable data-based bandwidth
#'   selection method for kernel density estimation. Journal of the Royal
#'   Statistical Society, Series B, 53, 683–690.
#'
#' @examples
#' ## For reproducibility
#' set.seed(0)
#'
#' ## unbounded data
#' x <- rnorm(100)                    # simulate data
#' fit <- kde1d(x)                    # estimate density
#' dkde1d(1000, fit)                  # evaluate density estimate
#' summary(fit)                       # information about the estimate
#' plot(fit)                          # plot the density estimate
#' curve(dnorm(x), add = TRUE,        # add true density
#'       col = "red")
#'
#' ## bounded data, log-linear
#' x <- rgamma(100, shape = 1)        # simulate data
#' fit <- kde1d(x, xmin = 0, deg = 1) # estimate density
#' dkde1d(1000, fit)                  # evaluate density estimate
#' summary(fit)                       # information about the estimate
#' plot(fit)                          # plot the density estimate
#' curve(dgamma(x, shape = 1),        # add true density
#'       add = TRUE, col = "red",
#'       from = 1e-3)
#'
#' ## discrete data
#' x <- rbinom(100, size = 5, prob = 0.5)  # simulate data
#' x <- ordered(x, levels = 0:5)           # declare as ordered
#' fit <- kde1d(x)                         # estimate density
#' dkde1d(2, fit)                          # evaluate density estimate
#' summary(fit)                            # information about the estimate
#' plot(fit)                               # plot the density estimate
#' points(ordered(0:5, 0:5),               # add true density
#'        dbinom(0:5, 5, 0.5), col = "red")
#'
#' @importFrom cctools cont_conv
#' @importFrom stats na.omit
#' @export
kde1d <- function(x, xmin = NaN, xmax = NaN, mult = 1, bw = NA, deg = 2) {
    x <- na.omit(x)
    # sanity checks
    check_arguments(x, mult, xmin, xmax, bw, deg)

    # jittering for discrete variables
    x <- cctools::cont_conv(x)

    # bandwidth selection
    bw <- select_bw_cpp(boundary_transform(x, xmin, xmax), bw, mult,
                        length(attr(x, "i_disc")) == 1)

    # fit model
    fit <- fit_kde1d_cpp(x, bw, xmin, xmax, deg)

    # add info
    fit$jitter_info <- attributes(x)
    fit$var_name <- colnames(x)
    fit$nobs <- length(x)

    # return as kde1d object
    class(fit) <- "kde1d"
    fit
}

#' check and pre-process arguments passed to kde1d()
#' @noRd
check_arguments <- function(x, mult, xmin, xmax, bw, deg) {
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

    if (!(deg %in% 0:2))
        stop("deg must be either 0, 1, or 2.")
}

#' adjusts observations and evaluation points for boundary effects
#' @importFrom stats qnorm
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

