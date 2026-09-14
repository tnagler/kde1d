## Purpose

This release fixes the valgrind error reported for kde1d 1.2.0 on the CRAN
memory-test machines. `remove_nans()` called `conservativeResize(0)` when every
observation was dropped, which reaches `realloc(ptr, 0)`; the vectors are now
swapped for empty ones instead. The path is reached by zero-inflated fits whose
observations are all zero, which the test suite covers.

The version number skips 1.2.1 so that it matches the pinned `kde1d-cpp`
backend release.

## Test environments
* ubuntu 22.04 on (devel, release, oldrel)
* macOS (release)
* Windows server 2022 (release)
* CRAN win-builder (devel)

## R CMD check results

0 errors | 0 warnings | 0 notes

## Reverse dependencies

This patch changes no R-level or C++ API, only the memory handling inside a
backend helper, so 'rvinecopulib' and 'vinereg' are unaffected. Both are
checked by the repository's revdepcheck workflow.
