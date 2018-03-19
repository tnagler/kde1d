#include <RcppEigen.h>
#include "interpolation.hpp"
#include "lpdens.hpp"

// [[Rcpp::export]]
Eigen::VectorXd interpolate(Eigen::VectorXd x, Eigen::VectorXd values, Eigen::VectorXd grid)
{
    InterpolationGrid1d o(grid, values, 1);
    return o.interpolate(x);
}

// [[Rcpp::export]]
Eigen::MatrixXd kde(Eigen::VectorXd x, Eigen::VectorXd values, double bw)
{
    LPDens1d fit(x, 100, bw);
    // std::cout << "fitted" << std::endl;
    Eigen::MatrixXd out(values.size(), 3);
    out.col(0) = fit.d(values);
    out.col(1) = fit.p(values);
    out.col(2) = fit.q(values);
    return out;
}