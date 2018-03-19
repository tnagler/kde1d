#include <RcppEigen.h>
#include "interpolation.hpp"

// [[Rcpp::export]]
Eigen::VectorXd interpolate(Eigen::VectorXd x, Eigen::VectorXd values, Eigen::VectorXd grid)
{
    InterpolationGrid1d o(grid, values, 0);
    return o.interpolate(x);
}