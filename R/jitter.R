#' Conditionally equidistant jittering
#'
#' Converts ordered variables to numeric and Adds deterministic uniform noise.
#' See *Details*.
#'
#' Jittering makes discrete variables continuous by adding noise. This simple
#' trick allows to consistently estimate densities with tools designed for the
#' continuous case (see, Nagler, 2018a/b). The drawback is that estimates are
#' random and the noise may deteriorate the estimate by chance.
#'
#' Here, we add a form of deterministic noise that makes estimators well
#' behaved. Tied occurences of a factor level are spread out uniformly
#' (i.e., equidistantly) on the interval \eqn{[-0.5, 0.5]}. This is similar to
#' adding random noise that is uniformly distributed, conditional on the
#' observed outcome. Integrating over the outcome, one can check that the
#' unconditional noise distribution is also uniform on \eqn{[-0.5, 0.5]}.
#'
#' Asymptotically, the deterministic jittering variant is equivalent to the
#' random one.
#'
#' @param x observations; the function does nothing if `x` is already numeric.
#'
#' @export
#'
#' @references
#' Nagler, T. (2018a). *A generic approach to nonparametric function estimation
#' with mixed data.* Statistics & Probability Letters, 137:326–330,
#' [arXiv:1704.07457](https://arxiv.org/abs/1704.07457)
#'
#' Nagler, T. (2018b). *Asymptotic analysis of the jittering kernel density
#' estimator.* Mathematical Methods of Statistics, in press,
#' [arXiv:1705.05431](https://arxiv.org/abs/1705.05431)
#'
#' @examples
#' x <- as.factor(rbinom(10, 1, 0.5))
#' equi_jitter(x)
equi_jitter <- function(x) {
  if (is.numeric(x))
    return(x)
  x <- as.numeric(x)
  tab <- table(x)
  noise <- unname(unlist(lapply(tab, function(l) -0.5 + 1:l / (l + 1))))
  s <- sort(x, index.return = TRUE)
  (s$x + noise)[rank(x, ties.method = "first", na.last = "keep")]
}
