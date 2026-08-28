# kde1d 1.1.2

DEPENDENCIES

* The `kde1d-cpp` backend is now maintained as a pinned Git submodule while
  preserving the existing public include paths for downstream packages.

* Advanced the pinned `kde1d-cpp` backend to include its latest numerical and
  performance improvements.

NEW FEATURES

* The standalone C++ API now allows configuring the interpolation grid size.

* Added `boundary_repair` to control data-adaptive local-linear estimates at
  finite support endpoints. One-sided fits now use a scale-equivariant Box-Cox
  transformation with power parameter 1/4 instead of the previous log
  transformation; two-sided fits retain the probit transformation.

* Finite bounds now apply to discrete supports and to the continuous component
  of zero-inflated fits. Discrete bounds are integer levels and are adjusted by
  half a unit when fitting the jitter density.

* Density and distribution evaluation now reuse cached spline coefficients,
  cumulative integrals, and cell lookups. Quantiles invert these cached
  integrals directly, substantially speeding up `qkde1d()` and `rkde1d()`.

BUG FIXES

* Automatic bandwidth selection is run again when a standalone C++ estimator
  is refitted.

* Fixed weighted and zero-inflated log-likelihood and effective degrees of
  freedom calculations in the standalone C++ backend.

* Initialized metadata for models constructed from an interpolation grid and
  avoided division by zero for empty weighted bins.

* Fixed binning and interpolation behavior at right endpoints, aligned
  right-boundary influence values, resolved transformed grids across bounded
  supports, stabilized negligible FFT tail values, truncated densities outside
  finite support bounds, and kept discrete CDF values inside the unit interval.

* Restored scale equivariance for one-sided bounded estimates by scaling the
  boundary transformation offset in the units of the fitted data, and improved
  numerical behavior in boundary tails by using exact transformation
  Jacobians.

* Fixed quantile evaluation for fully zero-inflated estimates and improved
  quantile accuracy at spline-cell boundaries and in flat cells.


# kde1d 1.1.1

BUG FIX

* Fix template deduction ambiguity in pre-CXX17 compilers.



# kde1d 1.1.0

NEW FEATURES

* Added functionality for estimating zero-inflated discrete-continuous mixtures.

* New `kde1d(..., type = "...")` argument to specify the data type. Options are 
  {c, cont, continuous} for continuous variables, {d, disc, discrete} for 
  discrete integer variables, or {zi, zinfl, zero-inflated} for zero-inflated
  variables.

BREAKING CHANGE

* New C++ API, making  it easier to use stand-alone.


# kde1d 1.0.7

DEPENDS

  * stop enforcing C++11.
  
# kde1d 1.0.5

BUG FIXES

  * fix cdf and input checks with NAs.

  
# kde1d 1.0.4

BUG FIXES

  * avoid bit-wise operations on Boolean variables.
  

# kde1d 1.0.3

BUG FIXES

  * fix invisible output in `dkde1d()`.
  

# kde1d 1.0.2

BUG FIXES

  * Prevent false positive on valgrind.


# kde1d 1.0.1

DEPENDENCIES

  * Removed dependence on `qrng` (#46).

BUG FIXES

  * Fixed undefined behavior with potential to cause memory issues (#46).
  
  * Prevent rare `bw_ == NaN` cases (#46).
  
  * Fixed compiler warnings due to unused or uninitialized variables (#46).


# kde1d 1.0.0

DEPENDENCIES

  * removed dependency on `cctools`.

NEW FEATURES

  * optimal plug-in bandwidth selection for all polynomial degrees (#38).
  
  * avoid randomness through simplified, deterministic jittering, see 
    `equi_jitter()` (#40).
  
  * headers in `inst/include` can be used as standalone C++ library with 
    convenience wrappers for R (#41).
    
  * (several times) faster `pkde1d()`, `qkde1d()`, and `rkde1d()` due to
    a more clever algorithm for numerical integration (#42).
    
  * faster `kde1d()` thanks to the Fast Fourier Transform (#43).
  
BUG FIXES

  * improvements to numerical stability, inter- and extrapolation (#32, #35, 
  #37).


# kde1d 0.4.0

NEW FEATURE

  * allow weights for observations via `kde1d(..., weights = )` (#29).

BUG FIX

  * stabilized bandwidth selection in presence of ties and outliers.

  * keep debug symbols on Linux systems (following a request by Prof. Ripley).


# kde1d 0.2.0

NEW FEATURES

  * improved stability of density estimates near a boundary (#21).

BUG FIXES

  * consistent behavior when `dkde1d()` and `pkde1d()` are called with 
    non-`ordered` input although data are discrete (#19).
  
  * fixed bug in computation of kernel density estimates (#20).
  
  * adapt minimum `bw` allowed for discrete data to truncated Gaussian kernel 
    (#20).


# kde1d 0.1.2

NEW FEATURES

  * Faster interpolation using binary search to find cells (#17).

BUG FIXES

  * Fixed heap buffer overflows in interpolation routines (#15, #16).
  
  * Fixed bounds of cdf for fit discrete data when long doubles are not 
    supported (#16).


# kde1d 0.1.0

* Initial release.
