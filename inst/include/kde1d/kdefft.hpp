#pragma once

#include "stats.hpp"
#include <unsupported/Eigen/FFT>

namespace kde1d {

namespace fft {

//! Bandwidth selection for local-likelihood density estimation.
//! Methodology is similar to Sheather and Jones(1991), but asymptotic
//! bias/variance expressions are adapted for higher-order polynomials and
//! nearest neighbor bandwidths.
class KdeFFT
{
public:
  KdeFFT(const Eigen::VectorXd& x,
         double bw,
         const Eigen::VectorXd& weights = Eigen::VectorXd());

  Eigen::VectorXd kde_drv(size_t drv) const;
  Eigen::VectorXd get_bin_counts() const { return bin_counts_; };
  void set_bw(double bw) { bw_ = bw; };

private:
  Eigen::VectorXd linbin(const Eigen::VectorXd& x,
                         double lower,
                         double upper,
                         const Eigen::VectorXd& weights) const;

  double bw_;
  Eigen::VectorXd bin_counts_;
  double lower_;
  double upper_;
  size_t num_bins_{ 400 };
};

//! @param x vector of observations.
//! @param weigths optional vector of weights for each observation.
inline KdeFFT::KdeFFT(const Eigen::VectorXd& x,
                      double bw,
                      const Eigen::VectorXd& weights)
{
  if (weights.size() > 0 && (weights.size() != x.size()))
    throw std::runtime_error("x and weights must have the same size");

  Eigen::VectorXd w;
  if (weights.size() > 0) {
    w = weights / weights.mean();
  } else {
    w = Eigen::VectorXd::Ones(x.size());
  }

  lower_ = x.minCoeff() - 3 * bw;
  upper_ = x.maxCoeff() + 3 * bw;
  bin_counts_ = linbin(x, lower_, upper_, w);
}

//! Computes bin counts for univariate data via the linear binning strategy.
//! @param x vector of observations
//! @param weights vector of weights for each observation.
inline Eigen::VectorXd
KdeFFT::linbin(const Eigen::VectorXd& x,
               double lower,
               double upper,
               const Eigen::VectorXd& weights) const
{
  Eigen::VectorXd gcnts = Eigen::VectorXd::Zero(num_bins_ + 1);
  double rem, lxi, delta;

  delta = (upper_ - lower_) / num_bins_;
  for (size_t i = 0; i < x.size(); ++i) {
    lxi = (x(i) - lower_) / delta;
    size_t li = static_cast<size_t>(lxi);
    rem = lxi - li;
    if (li < num_bins_) {
      gcnts(li) += (1 - rem) * weights(i);
      gcnts(li + 1) += rem * weights(i);
    }
  }

  return gcnts;
}

//! Binned kernel density derivative estimate
//! @param drv order of derivative.
//! @return estimated derivative evaluated at the bin centers.
inline Eigen::VectorXd
KdeFFT::kde_drv(size_t drv) const
{
  double delta = (upper_ - lower_) / num_bins_;
  double tau = 4.0 + drv;
  size_t L = std::floor(tau * bw_ / delta);
  L = std::min(L, num_bins_);

  double tmp_dbl = L * delta / bw_;
  Eigen::VectorXd arg = Eigen::VectorXd::LinSpaced(L + 1, 0.0, tmp_dbl);
  tmp_dbl = std::pow(bw_, drv + 1.0);
  arg = stats::dnorm_drv(arg, drv) / (tmp_dbl * bin_counts_.sum());

  tmp_dbl = num_bins_ + L + 1.0;
  tmp_dbl = std::pow(2, std::ceil(std::log(tmp_dbl) / std::log(2)));
  size_t P = static_cast<size_t>(tmp_dbl);

  Eigen::VectorXd arg2 = Eigen::VectorXd::Zero(P);
  arg2.head(L + 1) = arg;
  arg2.tail(L) = arg.tail(L).reverse() * (drv % 2 ? -1.0 : 1.0);

  Eigen::VectorXd x2 = Eigen::VectorXd::Zero(P);
  x2.head(num_bins_) = bin_counts_;

  Eigen::FFT<double> fft;
  Eigen::VectorXcd tmp1 = fft.fwd(arg2);
  Eigen::VectorXcd tmp2 = fft.fwd(x2);
  tmp1 = tmp1.cwiseProduct(tmp2);
  tmp2 = fft.inv(tmp1);
  return tmp2.head(num_bins_).real();
}

} // end kde1d::bw

} // end kde1d
