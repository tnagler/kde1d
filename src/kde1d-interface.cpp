#include <RcppEigen.h>
#include "kde1d-wrappers.hpp"

using namespace kde1d;

//' fits a kernel density estimate and calculates the effective degrees of
//' freedom.
//' @param x vector of observations; categorical data must be converted to
//'   non-negative integers.
//' @param xmin lower bound for the support of the density, `NaN` means no
//'   boundary.
//' @param xmax upper bound for the support of the density, `NaN` means no
//'   boundary.
//' @param type variable type; must be one of {c, cont, continuous} for
//'   continuous variables, one of {d, disc, discrete} for discrete integer
//'   variables, or one of {zi, zinfl, zero-inflated} for zero-inflated
//'   variables.
//' @param bandwidth the bandwidth parameter.
//' @param mult positive bandwidth multiplier; the actual bandwidth used is
//'   bw*mult.
//' @param degree order of the local polynomial.
//' @return `An Rcpp::List` containing the fitted density values on a grid and
//'   additional information.
//' @noRd
// [[Rcpp::export]]
Rcpp::List fit_kde1d_cpp(const Eigen::VectorXd& x,
                         double xmin,
                         double xmax,
                         std::string type,
                         double mult,
                         double bandwidth,
                         size_t degree,
                         const Eigen::VectorXd& weights)
{
  Kde1d model(xmin, xmax, type, mult, bandwidth, degree);
  model.fit(x, weights);
  return kde1d_wrap(model);
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
  return kde1d_wrap(kde1d_r).pdf(x, false);
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
  return kde1d_wrap(kde1d_r).cdf(q, false);
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
  return kde1d_wrap(kde1d_r).quantile(p, false);
}

