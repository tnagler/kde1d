#' check if data violate boundaries
#' @noRd
check_boundary_violations <- function(x, xmin, xmax) {
  if (!is.nan(xmax) & !is.nan(xmin)) {
    if (any(x < xmin) || any(x > xmax)) {
      stop("Not all data are contained in the interval [xmin, xmax].")
    }
  } else if (!is.nan(xmin)) {
    if (any(x < xmin)) {
      stop("Not all data are larger than xmin.")
    }
  } else if (!is.nan(xmax)) {
    if (any(x > xmax)) {
      stop("Not all data are samller than xmax.")
    }
  }
}

#' check and pre-process arguments passed to kde1d()
#' @noRd
check_arguments <- function(x, mult, xmin, xmax, bw, deg, weights) {
  stopifnot(NCOL(x) == 1)
  stopifnot(length(mult) == 1)
  stopifnot(length(xmin) == 1)
  stopifnot(length(xmax) == 1)
  stopifnot(length(bw) == 1)
  stopifnot(length(deg) == 1)

  stopifnot(is.numeric(mult))
  stopifnot(mult > 0)
  stopifnot(is.numeric(xmin))
  stopifnot(is.numeric(xmax))
  stopifnot(is.numeric(xmax))
  stopifnot(is.na(bw) | (is.numeric(bw) & (bw > 0)))
  stopifnot(is.numeric(deg))

  if (!is.ordered(x) & is.factor(x)) {
    stop("Factors not allowed; use kdevine::kdevine() or cctools::cckde().")
  }

  if (is.ordered(x) & (!is.nan(xmin) | !is.nan(xmax))) {
    stop("xmin and xmax are not meaningful for x of type ordered.")
  }

  if (!is.nan(xmax) & !is.nan(xmin)) {
    if (xmin > xmax) {
      stop("xmin is larger than xmax.")
    }
  }
  check_boundary_violations(x, xmin, xmax)

  if (!(deg %in% 0:2)) {
    stop("deg must be either 0, 1, or 2.")
  }

  if ((length(weights) > 0) && (length(weights) != length(x))) {
    stop("x and weights must have same length.")
  }
}


#' prepares evaluation points  observations and evaluation points for boundary effects
#' @importFrom stats qnorm
#' @noRd
prep_eval_arg <- function(x, obj) {
  if (is.data.frame(x))
    x <- x[[1]]
  if (!is.ordered(x) & is.ordered(obj$x))
    x <- as.ordered(x)
  if (is.numeric(x))
    return(x)

  stopifnot(is.ordered(x))

  if (!all(levels(x) %in% levels(obj$x)))
    stop("'x' contains levels that weren't present when fitting.")
  levels(x) <- levels(obj$x)
  if (!is.ordered(x) & is.ordered(obj$x))
    x <- ordered(x, levels(obj$x))
  as.numeric(x) - 1
}
