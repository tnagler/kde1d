# kde1d 0.2.1

BUG FIXES

  * fix bug in computation of effective degrees of freedom when `deg = 2`.


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
