#pragma once

#include "kdefft.hpp"
#include "stats.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

namespace kde1d {

namespace bw {

//! Bandwidth selection for local-likelihood density estimation.
//! Methodology is similar to Sheather and Jones(1991), but asymptotic
//! bias/variance expressions are adapted for higher-order polynomials and
//! nearest neighbor bandwidths.
class PluginBandwidthSelector
{
public:
  PluginBandwidthSelector(const Eigen::VectorXd& x,
                          const Eigen::VectorXd& weights = Eigen::VectorXd());
  double select_bw(size_t deg);

private:
  double scale_est(const Eigen::VectorXd& x);
  double get_bw_for_bkfe(size_t drv);
  double ll_ibias2(size_t deg);
  double ll_ivar(size_t deg);

  fft::KdeFFT kde_;
  Eigen::VectorXd weights_;
  Eigen::VectorXd bin_counts_;
  double scale_;
};

//! @param x vector of observations.
//! @param weigths optional vector of weights for each observation.
inline PluginBandwidthSelector::PluginBandwidthSelector(
  const Eigen::VectorXd& x,
  const Eigen::VectorXd& weights)
  : kde_(fft::KdeFFT(x, 0.0, x.minCoeff(), x.maxCoeff(), weights))
  , weights_(weights)
{
  if (weights.size() == 0) {
    weights_ = Eigen::VectorXd::Ones(x.size());
  } else {
    weights_ = weights_ * x.size() / weights_.sum();
  }

  bin_counts_ = kde_.get_bin_counts();
  scale_ = scale_est(x);
}

//! Scale estimate (minimum of standard deviation and robust equivalent)
//! @param x vector of observations.
inline double
PluginBandwidthSelector::scale_est(const Eigen::VectorXd& x)
{
  double m_x = x.cwiseProduct(weights_).mean();
  Eigen::VectorXd sx = (x - Eigen::VectorXd::Constant(x.size(), m_x));
  double sd_x =
    std::sqrt(sx.cwiseAbs2().cwiseProduct(weights_).sum() / (x.size() - 1));
  Eigen::VectorXd q_x(2);
  q_x << 0.25, 0.75;
  q_x = stats::quantile(x, q_x, weights_);
  double scale = std::min((q_x(1) - q_x(0)) / 1.349, sd_x);
  if (scale == 0) {
    scale = (sd_x > 0) ? sd_x : 1.0;
  }
  return scale;
}

//! optimal bandwidths for kernel functionals (see Wand and Jones' book, 3.5)
//! only works for even drv
//! @param drv order of the derivative in the kernel functional.
inline double
PluginBandwidthSelector::get_bw_for_bkfe(size_t drv)
{
  if (drv % 2 != 0) {
    throw std::runtime_error("only even drv allowed.");
  }

  // effective sample size
  double n = std::pow(weights_.sum(), 2) / weights_.cwiseAbs2().sum();

  // start with normal reference rule (eq 3.7)
  int r = drv + 4;
  double psi = ((r / 2) % 2 == 0) ? 1 : -1;
  psi *= std::tgamma(r + 1);
  psi /= std::pow(2 * scale_, r + 1) * std::tgamma(r / 2 + 1) * std::sqrt(M_PI);
  double Kr = stats::dnorm_drv(Eigen::VectorXd::Zero(1), r - 2)(0);
  kde_.set_bw(std::pow(-2 * Kr / (psi * n), 1.0 / (r + 1)));

  // now use plug in to select the actual bandwidth (eq. 3.6)
  r -= 2;
  // that's bkfe()
  psi = bin_counts_.cwiseProduct(kde_.kde_drv(drv + 2)).sum();
  psi /= bin_counts_.sum();

  Kr = stats::dnorm_drv(Eigen::VectorXd::Zero(1), r - 2)(0);

  return std::pow(-2 * Kr / (psi * n), 1.0 / (r + 1));
}

//! computes the integrated squared bias (without bw and n terms).
//! Bias expressions can be found in Geenens (JASA, 2014)
//! @param deg degree of the local polynomial.
inline double
PluginBandwidthSelector::ll_ibias2(size_t deg)
{
  Eigen::VectorXd arg;
  if (deg == 0) {
    kde_.set_bw(get_bw_for_bkfe(4));
    arg = 0.25 * kde_.kde_drv(4);
  } else if (deg == 1) {
    kde_.set_bw(get_bw_for_bkfe(4));
    Eigen::VectorXd f0 = kde_.kde_drv(0);
    Eigen::VectorXd f1 = kde_.kde_drv(1);
    Eigen::VectorXd f2 = kde_.kde_drv(2);
    arg = (0.5 * f2 + f1.cwiseAbs2().cwiseQuotient(f0))
            .cwiseAbs2()
            .cwiseQuotient(f0);
  } else if (deg == 2) {
    kde_.set_bw(get_bw_for_bkfe(8));
    Eigen::VectorXd f0 = kde_.kde_drv(0);
    Eigen::VectorXd f1 = kde_.kde_drv(1);
    Eigen::VectorXd f2 = kde_.kde_drv(2);
    Eigen::VectorXd f4 = kde_.kde_drv(4);
    arg = f4 - 3 * f2.cwiseAbs2().cwiseQuotient(f0) +
          2 * (f1.array().pow(4) / f0.array().pow(3)).matrix();
    arg = (0.125 * arg).cwiseAbs2().cwiseQuotient(f0);
  } else {
    throw std::runtime_error("deg must be one of {0, 1, 2}.");
  }
  return bin_counts_.cwiseProduct(arg).sum() / bin_counts_.sum();
}

//! computes the integrated squared variance (without bw and n terms).
//! Variance expressions can be found in Geenens (JASA, 2014)
  //! @param deg degree of the local polynomial.
inline double
PluginBandwidthSelector::ll_ivar(size_t deg)
{
  if (deg > 2)
    throw std::runtime_error("deg must be one of {0, 1, 2}.");
  return (deg < 2 ? 1.0 : 27.0 / 16.0) * 0.5 / std::sqrt(M_PI);
}

//! Selects the bandwidth for kernel density estimation.
//! @param deg degree of the local polynomial.
inline double
PluginBandwidthSelector::select_bw(size_t deg)
{
  // effective sample size
  double n = std::pow(weights_.sum(), 2) / weights_.cwiseAbs2().sum();
  double bw;
  int bwpow = (deg < 2 ? 4 : 8);
  try {
    double ibias2 = ll_ibias2(deg);
    double ivar = ll_ivar(deg);
    bw = std::pow(ivar / (bwpow * n * ibias2), 1.0 / (bwpow + 1));
  } catch (...) {
    bw = 4.0 * 1.06 * scale_ * std::pow(n, -1.0 / (bwpow + 1));
  }

  return bw;
}

} // end kde1d::bw

} // end kde1d
