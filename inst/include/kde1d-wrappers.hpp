#pragma once

#include "kde1d/kde1d.hpp"

namespace kde1d {

inline Rcpp::List kde1d_wrap(const Kde1d& kde1d_cpp)
{
  const auto state = kde1d_cpp.get_state();
  auto kde1d_r = Rcpp::List::create(
    Rcpp::Named("grid_points") = kde1d_cpp.get_grid_points(),
    Rcpp::Named("values") = kde1d_cpp.get_values(),
    Rcpp::Named("xmin") = kde1d_cpp.get_xmin(),
    Rcpp::Named("xmax") = kde1d_cpp.get_xmax(),
    Rcpp::Named("type") = kde1d_cpp.get_type_str(),
    Rcpp::Named("bw") = state.bandwidth,
    Rcpp::Named("bw_spec") = state.bandwidth_spec,
    Rcpp::Named("mult") = state.multiplier,
    Rcpp::Named("deg") = state.degree,
    Rcpp::Named("grid_size") = state.grid_size,
    Rcpp::Named("boundary_repair") = state.boundary_repair,
    Rcpp::Named("prob0") = kde1d_cpp.get_prob0(),
    Rcpp::Named("edf") = state.edf,
    Rcpp::Named("loglik") = state.loglik
  );
  kde1d_r.attr("class") = "kde1d";

  return kde1d_r;
}

inline Kde1d kde1d_wrap(const Rcpp::List& kde1d_r)
{
  auto grid = interp::InterpolationGrid(
    kde1d_r["grid_points"], kde1d_r["values"], 0);
  Kde1dState state;
  state.multiplier = kde1d_r["mult"];
  state.bandwidth_spec = kde1d_r["bw_spec"];
  state.bandwidth = kde1d_r["bw"];
  state.degree = kde1d_r["deg"];
  state.grid_size = kde1d_r["grid_size"];
  state.boundary_repair = kde1d_r["boundary_repair"];
  state.edf = kde1d_r["edf"];
  state.loglik = kde1d_r["loglik"];
  std::string var_type = kde1d_r["type"];
  return Kde1d(grid, kde1d_r["xmin"], kde1d_r["xmax"],
               var_type, kde1d_r["prob0"], state);
}


}
