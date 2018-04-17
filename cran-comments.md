## New submission after package has been archived

Problem 1: ASAN failures 
Solution: Fixed heap buffer overflows on clang-ASAN related to max-comparison
of a size_t and a negative number.

Problem 2: Failed without long doubles
Solution: Fixed cdf calculations for discrete data which made /tests fail.

Problem 3: Threw a C++ exception under valgrind (fatal to the R process)
Solution: This is a known issue of valgrind + boost; fixed by adding
'#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false' in /inst/include/stats.hpp 
(as suggested in http://svn.boost.org/trac10/ticket/10005).

We are sorry for the delay in fixing these issues, reproducing the bugs reported 
by clang-ASAN was way harder than expected.

## Test environments
* Debian Linux with clang-4.0-ASAN enabled on rocker/r-devel-ubsan-clang (devel)
* Ubuntu 16.04 with clang-7.0.0-ASAN enabled (devel)
* Ubuntu 16.04 without long double support (devel)
* Ubuntu 16.04 with valgrind (devel)
* local OS X install (release)
* ubuntu 12.04 on travis-ci (oldrel, release, devel)
* win-builder (devel)
* Windows Server 2012 R2 x64 + x86 (release)

## R CMD check results

0 errors | 0 warnings | 0 notes

## Reverse dependencies

This is a new release, so there are no reverse dependencies.
