#pragma once

#include "kde1d/kde1d.hpp"

namespace kde1d {

inline Rcpp::List kde1d_wrap(const Kde1d& kde1d_cpp)
{
  auto kde1d_r = Rcpp::List::create(
    Rcpp::Named("grid_points") = kde1d_cpp.get_grid_points(),
    Rcpp::Named("values") = kde1d_cpp.get_values(),
    Rcpp::Named("nlevels") = kde1d_cpp.get_nlevels(),
    Rcpp::Named("bw") = kde1d_cpp.get_bw(),
    Rcpp::Named("xmin") = kde1d_cpp.get_xmin(),
    Rcpp::Named("xmax") = kde1d_cpp.get_xmax(),
    Rcpp::Named("deg") = kde1d_cpp.get_deg(),
    Rcpp::Named("edf") = kde1d_cpp.get_edf(),
    Rcpp::Named("loglik") = kde1d_cpp.get_loglik()
  );
  kde1d_r.attr("class") = "kde1d";

  return kde1d_r;
}

inline Kde1d kde1d_wrap(const Rcpp::List& kde1d_r)
{
  auto grid = interp::InterpolationGrid1d(
    kde1d_r["grid_points"], kde1d_r["values"], 0);
  return Kde1d(grid, kde1d_r["nlevels"], kde1d_r["xmin"], kde1d_r["xmax"]);
}


}