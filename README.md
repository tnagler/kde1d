# kde1d

[![Build status Linux](https://travis-ci.org/tnagler/kde1d.svg?branch=master)](https://travis-ci.org/tnagler/kde1d)
[![Windows Build status](http://ci.appveyor.com/api/projects/status/github/tnagler/kde1d?svg=true)](https://ci.appveyor.com/project/tnagler/kde1d)
[![CRAN version](http://www.r-pkg.org/badges/version/kde1d)](https://cran.r-project.org/package=kde1d) 
[![Coverage Status](https://img.shields.io/codecov/c/github/tnagler/kde1d/master.svg)](https://codecov.io/github/tnagler/kde1d?branch=master)


### Summary

- implements a univariate kernel density estimator that can handle
bounded and discrete data. 
- provides classical kernel density as well as log-linear and log-quadratic methods. 
- is highly efficient due to spline interpolation and a C++ backend.

For details, see the 
[API documentation](https://tnagler.github.io/VineCopula/).

### How to install

- the stable release from CRAN:

``` r
install.packages("kde1d")
```

- the latest development version:

``` r
# install.packages("devtools")
devtools::install_github("tnagler/kde1d")
```

### Examples

##### Unbounded data
``` r
x <- rnorm(100)                    # simulate data
fit <- kde1d(x)                    # estimate density
dkde1d(1000, fit)                  # evaluate density estimate
summary(fit)                       # information about the estimate
plot(fit)                          # plot the density estimate
curve(dnorm(x), add = TRUE,        # add true density
      col = "red")
```

##### Bounded data, log-linear
``` r
x <- rgamma(100, shape = 1)        # simulate data
fit <- kde1d(x, xmin = 0, deg = 1) # estimate density
dkde1d(1000, fit)                  # evaluate density estimate
summary(fit)                       # information about the estimate
plot(fit)                          # plot the density estimate
curve(dgamma(x, shape = 1),        # add true density
      add = TRUE, col = "red",
      from = 1e-3)
```

##### Discrete data
``` r
x <- rbinom(100, size = 5, prob = 0.5)  # simulate data
x <- ordered(x, levels = 0:5)           # declare as ordered
fit <- kde1d(x)                         # estimate density
dkde1d(2, fit)                          # evaluate density estimate
summary(fit)                            # information about the estimate
plot(fit)                               # plot the density estimate
points(ordered(0:5, 0:5),               # add true density
       dbinom(0:5, 5, 0.5), col = "red")
```

### References

Geenens, G. (2014). *Probit transformation for kernel density estimation on
the unit interval*. Journal of the American Statistical Association,
109:505, 346-358, [arXiv:1303.4121](https://arxiv.org/abs/1303.4121)

Geenens, G., Wang, C. (2018). *Local-likelihood transformation kernel
density estimation for positive random variables.* Journal of Computational
and Graphical Statistics, to appear,
[arXiv:1602.04862](https://arxiv.org/abs/1602.04862)

Loader, C. (2006). *Local regression and likelihood*. Springer Science & 
Business Media.

Nagler, T. (2018a). *A generic approach to nonparametric function
estimation with mixed data.* Statistics & Probability Letters, 137:326–330,
[arXiv:1704.07457](https://arxiv.org/abs/1704.07457)

Nagler, T. (2018b). *Asymptotic analysis of the jittering kernel density
estimator.* Mathematical Methods of Statistics, in press,
[arXiv:1705.05431](https://arxiv.org/abs/1705.05431)
