#' Univariate local-polynomial likelihood kernel density estimation
#'
#' The estimators can handle data with bounded, unbounded, and discrete support,
#' see *Details*.
#'
#' @param x vector (or one-column matrix/data frame) of observations; can be
#'   `numeric` or `ordered`.
#' @param xmin lower support bound; `NaN` means no boundary. For discrete data,
#'   this is the lowest possible integer level. For zero-inflated data, it
#'   bounds only the continuous component, so zeros are exempt.
#' @param xmax upper support bound; `NaN` means no boundary. For discrete data,
#'   this is the highest possible integer level. For zero-inflated data, it
#'   bounds only the continuous component.
#' @param type variable type; must be one of `{c, cont, continuous}` for
#'   continuous variables, one of `{d, disc, discrete}` for discrete integer
#'   variables, or one of `{zi, zinfl, zero-inflated, zero_inflated}` for
#'   zero-inflated variables.
#' @param mult positive bandwidth multiplier; the actual bandwidth used is
#'   \eqn{bw*mult}.
#' @param bw bandwidth parameter; has to be a positive number or `NA`; the
#'   latter uses the plug-in methodology of Sheather and Jones (1991) with
#'   appropriate modifications for `deg > 0`.
#' @param deg degree of the polynomial; either `0`, `1`, or `2` for
#'   log-constant, log-linear, and log-quadratic fitting, respectively.
#' @param weights optional vector of weights for individual observations.
#' @param boundary_repair whether finite support endpoints are eligible for a
#'   data-adaptive boundary estimate. The default is `TRUE`; `FALSE` uses the
#'   transformed estimate throughout.
#'
#' @return An object of class `kde1d`.
#'
#' @details A Gaussian kernel is used in all cases. With one finite endpoint,
#'   the continuous component is fitted after an endpoint-anchored, scaled
#'   Box-Cox transformation with power parameter \eqn{1/4}. With two finite
#'   endpoints, it uses a regularized probit transformation (Geenens, 2014).
#'
#'   If `boundary_repair = TRUE`, each finite endpoint is classified from the
#'   observed tail. Endpoints consistent with a finite nonzero limiting density
#'   receive a nonnegative local-linear boundary estimate, which is fused with
#'   the transformed estimate using shrinking biweight weights. Other endpoints
#'   retain the transformed estimate. The result is normalized on the original
#'   scale.
#'
#'   Discrete variables are handled via deterministic jittering (Nagler, 2018a,
#'   2018b), see [equi_jitter()]. Finite integer bounds are shifted outward by
#'   one half before applying the same support transformations and boundary
#'   repair to the jitter density.
#'
#'   Zero-inflated densities are estimated by a hurdle-model with discrete
#'   mass at 0 and the nonzero observations estimated as a continuous
#'   component. Finite bounds constrain this component only.
#'
#' @seealso [`dkde1d()`], [`pkde1d()`], [`qkde1d()`], [`rkde1d()`],
#'   [`plot.kde1d()`], [`lines.kde1d()`]
#'
#' @references Geenens, G. (2014). *Probit transformation for kernel density
#' estimation on the unit interval*. Journal of the American Statistical
#' Association, 109:505, 346-358,
#' [arXiv:1303.4121](https://arxiv.org/abs/1303.4121)
#'
#' Box, G. E. P., Cox, D. R. (1964). *An analysis of transformations.* Journal
#' of the Royal Statistical Society, Series B, 26, 211--252.
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
#' fit <- kde1d(x, xmin = 0, xmax = 5, type = "discrete") # estimate density
#' fit <- kde1d(ordered(x, levels = 0:5)) # alternative API
#' dkde1d(sort(unique(x)), fit) # evaluate density estimate
#' summary(fit) # information about the estimate
#' plot(fit) # plot the density estimate
#' points(ordered(0:5, 0:5), # add true density
#'   dbinom(0:5, 5, 0.5),
#'   col = "red"
#' )
#'
#' ## zero-inflated data
#' x <- rexp(500, 0.5)  # simulate data
#' x[sample(1:500, 200)] <- 0 # add zero-inflation
#' fit <- kde1d(x, xmin = 0, type = "zi") # estimate density
#' dkde1d(sort(unique(x)), fit) # evaluate density estimate
#' summary(fit) # information about the estimate
#' plot(fit) # plot the density estimate
#' lines(  # add true density
#'   seq(0, 20, l = 100),
#'   0.6 * dexp(seq(0, 20, l = 100), 0.5),
#'   col = "red"
#' )
#' points(0, 0.4, col = "red")
#'
#' ## weighted estimate
#' x <- rnorm(100) # simulate data
#' weights <- rexp(100) # weights as in Bayesian bootstrap
#' fit <- kde1d(x, weights = weights) # weighted fit
#' plot(fit) # compare with unweighted fit
#' lines(kde1d(x), col = 2)
#' @importFrom stats na.omit
#' @export
kde1d <- function(x, xmin = NaN, xmax = NaN, type = "continuous",
                  mult = 1, bw = NA, deg = 2, weights = numeric(0),
                  boundary_repair = TRUE) {

  if (is.ordered(x)) {
    type <- "discrete"
    xmin <- 0
    xmax <- nlevels(x) - 1
  }

  # fit model
  fit <- fit_kde1d_cpp(x = if (is.numeric(x)) x else (as.numeric(x) - 1),
                       xmin = xmin,
                       xmax = xmax,
                       type = type,
                       bandwidth = bw,
                       mult = mult,
                       degree = deg,
                       weights = weights,
                       boundary_repair = boundary_repair)

  # add info
  fit$x <- x
  fit$weights <- weights
  fit$nobs <- sum(!is.na(x))
  fit$var_name <- as.character(match.call()[2])

  fit
}
