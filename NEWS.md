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
