#' check if data violate boundaries
#' @noRd
check_boundary_violations <- function(x, xmin, xmax) {
    if (!is.nan(xmax) & !is.nan(xmin)) {
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

#' check and pre-process arguments passed to kde1d()
#' @noRd
check_arguments <- function(x, mult, xmin, xmax, bw, nn, deg, weights) {
    stopifnot(NCOL(x) == 1)
    stopifnot(length(mult) == 1)
    stopifnot(length(xmin) == 1)
    stopifnot(length(xmax) == 1)
    stopifnot(length(bw) == 1)
    stopifnot(length(nn) == 1)
    stopifnot(length(deg) == 1)

    stopifnot(is.numeric(mult))
    stopifnot(mult > 0)
    stopifnot(is.numeric(xmin))
    stopifnot(is.numeric(xmax))
    stopifnot(is.numeric(xmax))
    stopifnot(is.na(bw) | (is.numeric(bw) & (bw > 0)))
    stopifnot(is.na(nn) | (is.numeric(nn) & (nn >= 0) & (nn <= 1)))
    stopifnot(is.numeric(deg))

    if (!is.ordered(x) & is.factor(x))
        stop("Factors not allowed; use kdevine::kdevine() or cctools::cckde().")

    if (is.ordered(x) & (!is.nan(xmin) | !is.nan(xmax)))
        stop("xmin and xmax are not meaningful for x of type ordered.")

    if (!is.nan(xmax) & !is.nan(xmin)) {
        if (xmin > xmax)
            stop("xmin is larger than xmax.")
    }
    check_boundary_violations(x, xmin, xmax)

    if (!(deg %in% 0:2))
        stop("deg must be either 0, 1, or 2.")

    if ((length(weights) > 0) && (length(weights) != length(x)))
        stop("x and weights must have same length.")
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

#' adjusts observations and evaluation points for boundary effects
#' @importFrom stats optim
#' @noRd
select_nn <- function(x, bw, xmin, xmax, deg, weights) {
    crit <- function(nn) {
        e <- try(fit <- fit_kde1d_cpp(x, bw, nn, xmin, xmax, deg, weights))
        if (inherits(2, "try-error"))
            return(Inf)
        -2 * fit$loglik + 2 * fit$edf
    }
    opt <- optim(0.5, crit, lower = 0, upper = 1, method = "Brent",
                 control = list(ndeps = 1e-2, reltol = 1e-2, maxit = 10))
    opt$par
}