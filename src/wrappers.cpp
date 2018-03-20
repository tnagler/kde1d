#include <RcppEigen.h>
#include "lpdens.hpp"

// [[Rcpp::export]]
Rcpp::List fit_kde1d_cpp(const Eigen::VectorXd& x,
                         double bw,
                         double xmin,
                         double xmax)
{
    LPDens1d fit(x, bw, xmin, xmax);
    return Rcpp::List::create(
        Rcpp::Named("grid_points") = fit.get_grid_points(),
        Rcpp::Named("values") = fit.get_values(),
        Rcpp::Named("bw") = bw,
        Rcpp::Named("xmin") = xmin,
        Rcpp::Named("xmax") = xmax,
        Rcpp::Named("edf") = fit.get_edf(),
        Rcpp::Named("loglik") = fit.get_loglik()
    );
}

InterpolationGrid1d wrap_to_cpp(const Rcpp::List& R_object)
{
    Eigen::VectorXd grid_points = R_object["grid_points"];
    Eigen::VectorXd values = R_object["values"];
    return InterpolationGrid1d(grid_points, values, 0);
}

// [[Rcpp::export]]
Eigen::VectorXd dkde1d_cpp(const Eigen::VectorXd& x,
                           const Rcpp::List& R_object)
{
    return wrap_to_cpp(R_object).interpolate(x);
}

// [[Rcpp::export]]
Eigen::VectorXd pkde1d_cpp(const Eigen::VectorXd& x,
                           const Rcpp::List& R_object)
{
    return wrap_to_cpp(R_object).integrate(x);
}

// [[Rcpp::export]]
Eigen::VectorXd qkde1d_cpp(const Eigen::VectorXd& x,
                           const Rcpp::List& R_object)
{
    InterpolationGrid1d fit = wrap_to_cpp(R_object);
    auto f = [&fit] (const Eigen::VectorXd& xx) {
        return fit.integrate(xx);
    };

    return invert_f(x,
                    f,
                    fit.get_grid_points().minCoeff(),
                    fit.get_grid_points().maxCoeff(),
                    20);
}