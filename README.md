# kde1d

<!-- badges: start -->

[![R build
status](https://github.com/tnagler/kde1d/workflows/R-CMD-check/badge.svg)](https://github.com/tnagler/kde1d/actions)
[![CRAN
version](http://www.r-pkg.org/badges/version/kde1d)](https://cran.r-project.org/package=kde1d)
[![Coverage
Status](https://img.shields.io/codecov/c/github/tnagler/kde1d/master.svg)](https://codecov.io/github/tnagler/kde1d?branch=master)
<!-- badges: end -->

### Summary

-   implements a univariate kernel density estimator that can handle
    bounded and discrete data.
-   provides classical kernel density as well as log-linear and
    log-quadratic methods.
-   is highly efficient due to the Fast Fourier Transform, spline
    interpolation, and a C++ backend.

For details, see the [API
documentation](https://tnagler.github.io/kde1d/).

### How to install

-   the stable release from CRAN:

<!-- -->

    install.packages("kde1d")

-   the latest development version:

<!-- -->

    # install.packages("remotes")
    remotes::install_github("tnagler/kde1d@dev")

### Examples

    library(kde1d)

##### Unbounded data

    x <- rnorm(100)                    # simulate data
    fit <- kde1d(x)                    # estimate density
    dkde1d(0, fit)                     # evaluate density estimate

    ## [1] 0.3783912

    summary(fit)                       # information about the estimate

    ## kernel density estimate ('kde1d'), log-quadratic
    ## -----------------------------------------------------------------
    ## nobs = 100, bw = 0.81, loglik = -132.63, d.f. = 2.78

    plot(fit)                          # plot the density estimate
    curve(dnorm(x), add = TRUE,        # add true density
          col = "red")

![](README_files/figure-markdown_strict/unnamed-chunk-2-1.png)

##### Bounded data, log-linear

    x <- rgamma(100, shape = 1)        # simulate data
    fit <- kde1d(x, xmin = 0, deg = 1) # estimate density
    dkde1d(seq(0, 5, by = 1), fit)     # evaluate density estimate

    ## [1] 0.790016219 0.416820849 0.135327122 0.044113159 0.013458525 0.003837033

    summary(fit)                       # information about the estimate

    ## kernel density estimate ('kde1d'), log-linear with bounded support (xmin = 0)
    ## -----------------------------------------------------------------
    ## nobs = 100, bw = 0.4, loglik = -110.24, d.f. = 7.38

    plot(fit)                          # plot the density estimate
    curve(dgamma(x, shape = 1),        # add true density
          add = TRUE, col = "red",
          from = 1e-3)

![](README_files/figure-markdown_strict/unnamed-chunk-3-1.png)

##### Discrete data

    x <- rbinom(100, size = 5, prob = 0.5)  # simulate data
    x <- ordered(x, levels = 0:5)           # declare as ordered
    fit <- kde1d(x)                         # estimate density
    dkde1d(sort(unique(x)), fit)            # evaluate density estimate

    ## [1] 0.02736751 0.15125898 0.30979502 0.33929591 0.15512435 0.01715822

    summary(fit)                            # information about the estimate

    ## (jittered) kernel density estimate ('kde1d'), log-quadratic
    ## -----------------------------------------------------------------
    ## nobs = 100, bw = 1.13, loglik = -145, d.f. = 11.24

    plot(fit)                               # plot the density estimate
    points(ordered(0:5, 0:5),               # add true density
           dbinom(0:5, 5, 0.5), col = "red")

![](README_files/figure-markdown_strict/unnamed-chunk-4-1.png)

##### Weighted estimate

    x <- rnorm(100)                         # simulate data
    weights <- rexp(100)                    # weights as in Bayesian bootstrap
    fit <- kde1d(x, weights = weights)      # weighted fit
    plot(fit)                               # compare with unweighted fit
    lines(kde1d(x), col = 2)

![](README_files/figure-markdown_strict/unnamed-chunk-5-1.png)

### References

Geenens, G. (2014). *Probit transformation for kernel density estimation
on the unit interval*. Journal of the American Statistical Association,
109:505, 346-358, [arXiv:1303.4121](https://arxiv.org/abs/1303.4121)

Geenens, G., Wang, C. (2018). *Local-likelihood transformation kernel
density estimation for positive random variables.* Journal of
Computational and Graphical Statistics, to appear,
[arXiv:1602.04862](https://arxiv.org/abs/1602.04862)

Loader, C. (2006). *Local regression and likelihood*. Springer Science &
Business Media.

Nagler, T. (2018a). *A generic approach to nonparametric function
estimation with mixed data.* Statistics & Probability Letters,
137:326–330, [arXiv:1704.07457](https://arxiv.org/abs/1704.07457)

Nagler, T. (2018b). *Asymptotic analysis of the jittering kernel density
estimator.* Mathematical Methods of Statistics, in press,
[arXiv:1705.05431](https://arxiv.org/abs/1705.05431)
