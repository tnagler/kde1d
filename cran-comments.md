Patch to prevent possible false positive on valgrind (only on some platforms):  https://cran.r-project.org/web/checks/check_results_kde1d.html

## Test environments
* ubuntu 18.04 with gcc7 and valgrind (release)
* ubuntu 18.04 clang 9 and valgrind (release)
* ubuntu 18.04 with clang ASAN/UBSAN on rocker (devel)
* ubuntu 14.04 on travis-ci (release, devel, oldrel)
* win-builder (devel, release)

## R CMD check results

0 errors | 0 warnings | 0 notes

## revdepcheck results

Checked 2 reverse dependencies (Note is unrelated to this package):

rvinecopulib 0.3.2.1.1    0 errors | 0 warnings | 1 note                                
vinereg 0.5.0             0 errors | 0 warnings | 0 notes                            
