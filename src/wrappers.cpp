#include <RcppEigen.h>
#include "dpik.hpp"
#include "lpdens.hpp"

//' fits a kernel density estimate and calculates the effective degrees of
//' freedom.
//' @param x vector of observations.
//' @param bw the bandwidth parameter.
//' @param xmin lower bound for the support of the density, `NaN` means no
//'   boundary.
//' @param xmax upper bound for the support of the density, `NaN` means no
//'   boundary.
//' @param deg order of the local polynomial.
//' @return `An Rcpp::List` containing the fitted density values on a grid and
//'   additional information.
//' @noRd
// [[Rcpp::export]]
Rcpp::List fit_kde1d_cpp(const Eigen::VectorXd& x,
                         double bw,
                         double xmin,
                         double xmax,
                         size_t deg)
{
    LPDens1d fit(x, bw, xmin, xmax, deg);
    return Rcpp::List::create(
        Rcpp::Named("grid_points") = fit.get_grid_points(),
        Rcpp::Named("values") = fit.get_values(),
        Rcpp::Named("bw") = bw,
        Rcpp::Named("xmin") = xmin,
        Rcpp::Named("xmax") = xmax,
        Rcpp::Named("deg") = deg,
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
    Eigen::VectorXd fhat = wrap_to_cpp(R_object).interpolate(x);
    double xmin = R_object["xmin"];
    double xmax = R_object["xmax"];
    if (!std::isnan(xmin)) {
        fhat = (x.array() < xmin).select(Eigen::VectorXd::Zero(x.size()), fhat);
    }
    if (!std::isnan(xmax)) {
        fhat = (x.array() > xmax).select(Eigen::VectorXd::Zero(x.size()), fhat);
    }

    return fhat;
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
    return wrap_to_cpp(R_object).integrate(x).array().max(0.0).min(1.0);
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

//  Bandwidth for Kernel Density Estimation
//' @param x vector of observations
//' @param grid_size number of equally-spaced points over which binning is
//' performed to obtain kernel functional approximation
//' @return the selected bandwidth
//' @noRd
// [[Rcpp::export]]
double select_bw_cpp(const Eigen::VectorXd& x,
                     double bw,
                     double mult,
                     bool discrete) {

    if (std::isnan(bw)) {
        bw = dpik(x);
    }

    bw *= mult;

    if (discrete) {
        bw = std::max(bw, 0.5 / 5);
    }

    return(bw);
}
