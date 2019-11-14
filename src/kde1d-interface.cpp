#include <RcppEigen.h>
#include "kde1d-wrappers.hpp"

using namespace kde1d;

//' fits a kernel density estimate and calculates the effective degrees of
//' freedom.
//' @param x vector of observations; catergorical data must be converted to
//'   non-negative integers.
//' @param nlevels the number of factor levels; 0 for continuous data.
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
                         size_t nlevels,
                         double bw,
                         double mult,
                         double xmin,
                         double xmax,
                         size_t deg,
                         const Eigen::VectorXd& weights)
{
  Kde1d fit(x, nlevels, bw, mult, xmin, xmax, deg, weights);
  return kde1d_wrap(fit);
}

//' computes the pdf of a kernel density estimate by interpolation.
//' @param x vector of evaluation points.
//' @param kde1d_r the fitted object passed from R.
//' @return a vector of pdf values.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dkde1d_cpp(const Eigen::VectorXd& x,
                           const Rcpp::List& kde1d_r)
{
  return kde1d_wrap(kde1d_r).pdf(x);
}

//' computes the cdf of a kernel density estimate by numerical integration.
//' @param x vector of evaluation points.
//' @param kde1d_r the fitted object passed from R.
//' @return a vector of cdf values.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd pkde1d_cpp(const Eigen::VectorXd& q,
                           const Rcpp::List& kde1d_r)
{
  return kde1d_wrap(kde1d_r).cdf(q);
}

//' computes the quantile of a kernel density estimate by numerical inversion
//' (bisection method).
//' @param x vector of evaluation points.
//' @param kde1d_r the fitted object passed from R.
//' @return a vector of quantiles.
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd qkde1d_cpp(const Eigen::VectorXd& p,
                           const Rcpp::List& kde1d_r)
{
  return kde1d_wrap(kde1d_r).quantile(p);
}

