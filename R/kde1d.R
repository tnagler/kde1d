#' Univariate local-polynomial likelihood kernel density estimation
#'
#' The estimators can handle data with bounded, unbounded, and discrete support,
#' see *Details*.
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
#'   latter uses the plug-in methodology of Sheather and Jones (1991) with
#'   appropriate modifications for `deg > 0`.
#'
#' @param deg degree of the polynomial; either `0`, `1`, or `2` for
#'   log-constant, log-linear, and log-quadratic fitting, respectively.
#' @param weights optional vector of weights for individual observations.
#'
#' @return An object of class `kde1d`.
#'
#' @details A gaussian kernel is used in all cases. If `xmin` or `xmax` are
#'   finite, the density estimate will be 0 outside of \eqn{[xmin, xmax]}. A
#'   log-transform is used if there is only one boundary (see, Geenens and Wang,
#'   2018); a probit transform is used if there are two (see, Geenens, 2014).
#'
#'   Discrete variables are handled via jittering (see, Nagler, 2018a, 2018b).
#'   A specific form of deterministic jittering is used, see [equi_jitter()].
#'
#'
#' @seealso [`dkde1d()`], [`pkde1d()`], [`qkde1d()`], [`rkde1d()`],
#'   [`plot.kde1d()`], [`lines.kde1d()`]
#'
#' @references Geenens, G. (2014). *Probit transformation for kernel density
#' estimation on the unit interval*. Journal of the American Statistical
#' Association, 109:505, 346-358,
#' [arXiv:1303.4121](https://arxiv.org/abs/1303.4121)
#'
#' Geenens, G., Wang, C. (2018). *Local-likelihood transformation kernel density
#' estimation for positive random variables.* Journal of Computational and
#' Graphical Statistics, to appear,
#' [arXiv:1602.04862](https://arxiv.org/abs/1602.04862)
#'
#' Nagler, T. (2018a). *A generic approach to nonparametric function estimation
#' with mixed data.* Statistics & Probability Letters, 137:326–330,
#' [arXiv:1704.07457](https://arxiv.org/abs/1704.07457)
#'
#' Nagler, T. (2018b). *Asymptotic analysis of the jittering kernel density
#' estimator.* Mathematical Methods of Statistics, in press,
#' [arXiv:1705.05431](https://arxiv.org/abs/1705.05431)
#'
#' Sheather, S. J. and Jones, M. C. (1991). A reliable data-based bandwidth
#' selection method for kernel density estimation. Journal of the Royal
#' Statistical Society, Series B, 53, 683–690.
#'
#' @examples
#'
#' ## unbounded data
#' x <- rnorm(500) # simulate data
#' fit <- kde1d(x) # estimate density
#' dkde1d(0, fit) # evaluate density estimate
#' summary(fit) # information about the estimate
#' plot(fit) # plot the density estimate
#' curve(dnorm(x),
#'   add = TRUE, # add true density
#'   col = "red"
#' )
#'
#' ## bounded data, log-linear
#' x <- rgamma(500, shape = 1) # simulate data
#' fit <- kde1d(x, xmin = 0, deg = 1) # estimate density
#' dkde1d(seq(0, 5, by = 1), fit) # evaluate density estimate
#' summary(fit) # information about the estimate
#' plot(fit) # plot the density estimate
#' curve(dgamma(x, shape = 1), # add true density
#'   add = TRUE, col = "red",
#'   from = 1e-3
#' )
#'
#' ## discrete data
#' x <- rbinom(500, size = 5, prob = 0.5) # simulate data
#' x <- ordered(x, levels = 0:5) # declare as ordered
#' fit <- kde1d(x) # estimate density
#' dkde1d(sort(unique(x)), fit) # evaluate density estimate
#' summary(fit) # information about the estimate
#' plot(fit) # plot the density estimate
#' points(ordered(0:5, 0:5), # add true density
#'   dbinom(0:5, 5, 0.5),
#'   col = "red"
#' )
#'
#' ## weighted estimate
#' x <- rnorm(100) # simulate data
#' weights <- rexp(100) # weights as in Bayesian bootstrap
#' fit <- kde1d(x, weights = weights) # weighted fit
#' plot(fit) # compare with unweighted fit
#' lines(kde1d(x), col = 2)
#' @importFrom stats na.omit
#' @export
kde1d <- function(x, xmin = NaN, xmax = NaN, mult = 1, bw = NA,  deg = 2,
                  weights = numeric(0)) {
  x <- na.omit(x)
  # sanity checks
  check_arguments(x, mult, xmin, xmax, bw, deg, weights)

  # fit model
  fit <- fit_kde1d_cpp(x = if (is.numeric(x)) x else (as.numeric(x) - 1),
                       nlevels = length(levels(x)),
                       bw = bw,
                       mult = mult,
                       xmin = xmin,
                       xmax = xmax,
                       deg = deg,
                       weights = weights)

  # add info
  fit$x <- x
  fit$weights <- weights
  fit$nobs <- sum(!is.na(x))
  fit$var_name <- as.character(match.call()[2])

  fit
}
