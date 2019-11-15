#' One-Dimensional Kernel Density Estimation
#'
#' Provides an efficient implementation of univariate local polynomial
#' kernel density estimators that can handle bounded and discrete data. The
#' implementation utilizes spline interpolation to reduce memory usage and
#' computational demand for large data sets.
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
#' @name kde1d-package
#' @docType package
NULL

#' @useDynLib kde1d
#' @importFrom Rcpp sourceCpp
NULL
