#include <RcppEigen.h>
#include "lpdens.hpp"

//' fits a kernel density estimate and calculates the effective degrees of
//' freedom.
//' @param x vector of observations.
//' @param bw the bandwidth parameter.
//' @param xmin lower bound for the support of the density, `NaN` means no
//'   boundary.
//' @param xmax upper bound for the support of the density, `NaN` means no
//'   boundary.
//' @return `An Rcpp::List` containing the fitted density values on a grid and
//'   additional information.
//' @noRd
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

// converts a fitted R_object ('kde1d') to an interpolation grid in C++.
// @param R_object the fitted object passed from R.
// @return C++ object of class InterpolationGrid1d.
InterpolationGrid1d wrap_to_cpp(const Rcpp::List& R_object)
{
    Eigen::VectorXd grid_points = R_object["grid_points"];
    Eigen::VectorXd values = R_object["values"];
    // 0 -> already normalized during fit
    return InterpolationGrid1d(grid_points, values, 0);
}

//' computes the pdf of a kernel density estimate by interpolation.
//' @param x vector of evaluation points.
//' @param R_object the fitted object passed from R.
//' @return a vector of pdf values.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dkde1d_cpp(const Eigen::VectorXd& x,
                           const Rcpp::List& R_object)
{
    return wrap_to_cpp(R_object).interpolate(x);
}

//' computes the cdf of a kernel density estimate by numerical integration.
//' @param x vector of evaluation points.
//' @param R_object the fitted object passed from R.
//' @return a vector of cdf values.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd pkde1d_cpp(const Eigen::VectorXd& x,
                           const Rcpp::List& R_object)
{
    return wrap_to_cpp(R_object).integrate(x);
}

//' computes the quantile of a kernel density estimate by numerical inversion
//' (bisection method).
//' @param x vector of evaluation points.
//' @param R_object the fitted object passed from R.
//' @return a vector of quantiles.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd qkde1d_cpp(const Eigen::VectorXd& x,
                           const Rcpp::List& R_object)
{
    InterpolationGrid1d fit = wrap_to_cpp(R_object);
    auto cdf = [&fit] (const Eigen::VectorXd& xx) {
        return fit.integrate(xx);
    };
    auto q = tools::invert_f(x,
                             cdf,
                             fit.get_grid_points().minCoeff(),
                             fit.get_grid_points().maxCoeff(),
                             35);

    // replace with NaN where the input was NaN
    for (size_t i = 0; i < x.size(); i++) {
        if (std::isnan(x(i)))
            q(i) = std::numeric_limits<double>::quiet_NaN();
    }

    return q;
}