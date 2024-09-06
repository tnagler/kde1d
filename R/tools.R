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
