// [[Rcpp::depends(kde1d, RcppEigen, BH)]]

#include <Rcpp.h>
#include <kde1d.hpp>
#include <kde1d/dpik.hpp>
#include <kde1d/interpolation.hpp>
#include <kde1d/kde1d.hpp>
#include <kde1d/kdefft.hpp>
#include <kde1d/stats.hpp>
#include <kde1d/tools.hpp>
#include <kde1d/version.hpp>

// [[Rcpp::export]]
bool kde1d_headers_compile()
{
  return true;
}
