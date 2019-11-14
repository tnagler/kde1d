#pragma once

#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false
#include <Eigen/Dense>
#include <algorithm>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <random>
#include <vector>
#include "tools.hpp"

namespace kde1d {

//! statistical functions
namespace stats {

//! standard normal density
//! @param x evaluation points.
//! @return matrix of pdf values.
inline Eigen::MatrixXd
dnorm(const Eigen::MatrixXd& x)
{
  boost::math::normal dist;
  return x.unaryExpr(
    [&dist](const double& y) { return boost::math::pdf(dist, y); });
};

//! standard normal density
//! @param x evaluation points.
//! @param drv order of the derivative
//! @return matrix of pdf values.
inline Eigen::MatrixXd
dnorm_drv(const Eigen::MatrixXd& x, unsigned drv)
{
  boost::math::normal dist;
  double rt2 = std::sqrt(2);
  return x.unaryExpr([&dist, &drv, &rt2](const double& y) {
    double res = boost::math::pdf(dist, y);
    // boost implementes phsyicist's hermite poly; rescale to probabilist's.
    res *= boost::math::hermite(drv, y / rt2);
    res *= std::pow(0.5, drv * 0.5);
    if (drv % 2)
      res = -res;
    return res;
  });
};

//! standard normal cdf
//! @param x evaluation points.
//! @return matrix of cdf values.
inline Eigen::MatrixXd
pnorm(const Eigen::MatrixXd& x)
{
  boost::math::normal dist;
  return x.unaryExpr(
    [&dist](const double& y) { return boost::math::cdf(dist, y); });
};

//! standard normal quantiles
//! @param x evaluation points.
//! @return matrix of quantiles.
inline Eigen::MatrixXd
qnorm(const Eigen::MatrixXd& x)
{
  boost::math::normal dist;
  return x.unaryExpr(
    [&dist](const double& y) { return boost::math::quantile(dist, y); });
};

//! empirical quantiles
//! @param x data.
//! @param q evaluation points.
//! @return vector of quantiles.
inline Eigen::VectorXd
quantile(const Eigen::VectorXd& x, const Eigen::VectorXd& q)
{
  double n = static_cast<double>(x.size() - 1);
  size_t m = q.size();
  Eigen::VectorXd res(m);

  // map to std::vector and sort
  std::vector<double> x2(x.data(), x.data() + x.size());
  std::sort(x2.begin(), x2.end());

  // linear interpolation (quantile of type 7 in R)
  for (size_t i = 0; i < m; ++i) {
    size_t k = std::floor(n * q(i));
    double p = static_cast<double>(k) / n;
    res(i) = x2[k];
    if (k < n)
      res(i) += (x2[k + 1] - x2[k]) * (q(i) - p) * n;
  }
  return res;
}

//! empirical quantiles
//! @param x data.
//! @param q evaluation points.
//! @param w vector of weights.
//! @return vector of quantiles.
inline Eigen::VectorXd
quantile(const Eigen::VectorXd& x,
         const Eigen::VectorXd& q,
         const Eigen::VectorXd& w)
{
  if (w.size() == 0)
    return quantile(x, q);
  if (w.size() != x.size())
    throw std::runtime_error("x and w must have the same size");
  double n = static_cast<double>(x.size());
  size_t m = q.size();
  Eigen::VectorXd res(m);

  // map to std::vector and sort
  std::vector<size_t> ind(n);
  for (size_t i = 0; i < n; ++i)
    ind[i] = i;
  std::sort(
    ind.begin(), ind.end(), [&x](size_t i, size_t j) { return x(i) < x(j); });

  auto x2 = x;
  auto wcum = w;
  auto wrank = Eigen::VectorXd::Constant(n, 0.0);
  double wacc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    x2(i) = x(ind[i]);
    wcum(i) = wacc;
    wacc += w(ind[i]);
  }

  double wsum = w.sum() - w(ind[n - 1]);
  ;
  for (size_t j = 0; j < m; ++j) {
    size_t i = 1;
    while ((wcum(i) < q(j) * wsum) & (i < n))
      i++;
    res(j) = x2(i - 1);
    if (w(ind[i - 1]) > 1e-30) {
      res(j) +=
        (x2(i) - x2(i - 1)) * (q(j) - wcum(i - 1) / wsum) / w(ind[i - 1]);
    }
  }

  return res;
}

// conditionally equidistant jittering; equivalent to the R implementation:
//   tab <- table(x)
//   noise <- unname(unlist(lapply(tab, function(l) -0.5 + 1:l / (l + 1))))
//   s <- sort(x, index.return = TRUE)
//   return((s$x + noise)[rank(x, ties.method = "first", na.last = "keep")])
inline Eigen::VectorXd
equi_jitter(const Eigen::VectorXd& x)
{
  size_t n = x.size();

  // first compute the corresponding permutation that sorts x (required later)
  auto perm = tools::get_order(x);
  // actually sort x
  Eigen::VectorXd srt(n);
  for (size_t i = 0; i < n; ++i)
    srt(i) = x(perm(i));

  // compute contingency table
  Eigen::MatrixXd tab(n + 1, 2);
  size_t lev = 0;
  size_t cnt = 1;
  for (size_t k = 1; k < n; ++k) {
    if (srt(k - 1) != srt(k)) {
      tab(lev, 0) = srt(k - 1);
      tab(lev++, 1) = cnt;
      cnt = 1;
    } else {
      cnt++;
      if (k == n - 1)
        tab(lev++, 1) = cnt;
    }
  }
  tab.conservativeResize(lev, 2);

  // add deterministic, conditionally uniorm noise
  Eigen::VectorXd noise(n);
  size_t i = 0;
  for (size_t k = 0; k < tab.rows(); ++k) {
    for (size_t cnt = 1; cnt <= tab(k, 1); ++cnt)
      noise(i++) = -0.5 + cnt / (tab(k, 1) + 1.0);
    cnt = 1;
  }
  Eigen::VectorXd jtr = srt + noise;

  // invert the permutation to return jittered x in original order
  for (size_t i = 0; i < perm.size(); ++i)
    srt(perm(i)) = jtr(i);

  return srt;
}

//! @brief simulates from the standard uniform distribution.
//!
//! @param n number of observations.
//! @param seeds seeds of the random number generator; if empty (default),
//!   the random number generator is seeded randomly.
//!
//! @return An size n vector of independent \f$ \mathrm{U}[0, 1] \f$ random
//!   variables.
inline Eigen::VectorXd
simulate_uniform(size_t n, std::vector<int> seeds)
{
  if (n < 1)
    throw std::runtime_error("n  must be at least 1.");

  if (seeds.size() == 0) { // no seeds provided, seed randomly
    std::random_device rd{};
    seeds = std::vector<int>(5);
    for (auto& s : seeds)
      s = static_cast<int>(rd());
  }

  // initialize random engine and uniform distribution
  std::seed_seq seq(seeds.begin(), seeds.end());
  std::mt19937 generator(seq);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  Eigen::VectorXd U(n);
  return U.unaryExpr([&](double) { return distribution(generator); });
}

} // end kde1d::stats

} // end kde1d
