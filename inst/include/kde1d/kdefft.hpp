#pragma once

#include "stats.hpp"
#include "tools.hpp"
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
         double bandwidth,
         double lower,
         double upper,
         const Eigen::VectorXd& weights = Eigen::VectorXd());

  Eigen::VectorXd kde_drv(unsigned drv) const;
  Eigen::VectorXd get_bin_counts() const { return bin_counts_; };
  void set_bandwidth(double bandwidth) { bandwidth_ = bandwidth; };

private:
  double bandwidth_;
  double lower_;
  double upper_;
  static constexpr size_t num_bins_{ 400 };
  Eigen::VectorXd bin_counts_;
};

//! @param x vector of observations.
//! @param bandwidth the bandwidth parameter.
//! @param lower lower bound of the grid.
//! @param upper bound of the grid.
//! @param weigths optional vector of weights for each observation.
inline KdeFFT::KdeFFT(const Eigen::VectorXd& x,
                      double bandwidth,
                      double lower,
                      double upper,
                      const Eigen::VectorXd& weights)
  : bandwidth_(bandwidth)
  , lower_(lower)
  , upper_(upper)
{
  if (weights.size() > 0 && (weights.size() != x.size()))
    throw std::invalid_argument("x and weights must have the same size");

  Eigen::VectorXd w;
  if (weights.size() > 0) {
    w = weights / weights.mean();
  } else {
    w = Eigen::VectorXd::Ones(x.size());
  }
  bin_counts_ = tools::linbin(x, lower_, upper_, num_bins_, w);
}

//! Binned kernel density derivative estimate
//! @param drv order of derivative.
//! @return estimated derivative evaluated at the bin centers.
inline Eigen::VectorXd
KdeFFT::kde_drv(unsigned drv) const
{
  double delta = (upper_ - lower_) / num_bins_;
  double tau = 4.0 + drv;
  size_t L = static_cast<size_t>(std::floor(tau * bandwidth_ / delta));
  L = std::min(L, num_bins_ + 1);

  double tmp_dbl = static_cast<double>(L) * delta / bandwidth_;
  Eigen::VectorXd arg = Eigen::VectorXd::LinSpaced(L + 1, 0.0, tmp_dbl);
  tmp_dbl = std::pow(bandwidth_, drv + 1.0);
  arg = stats::dnorm_drv(arg, drv) / (tmp_dbl * bin_counts_.sum());

  tmp_dbl = static_cast<double>(num_bins_ + L) + 2.0;
  tmp_dbl = std::pow(2, std::ceil(std::log(tmp_dbl) / std::log(2)));
  size_t P = static_cast<size_t>(tmp_dbl);

  Eigen::VectorXd arg2 = Eigen::VectorXd::Zero(P);
  arg2.head(L + 1) = arg;
  arg2.tail(L) = arg.tail(L).reverse() * (drv % 2 ? -1.0 : 1.0);

  Eigen::VectorXd x2 = Eigen::VectorXd::Zero(P);
  x2.head(num_bins_ + 1) = bin_counts_;

  Eigen::FFT<double> fft;
  Eigen::VectorXcd tmp1 = fft.fwd(arg2);
  Eigen::VectorXcd tmp2 = fft.fwd(x2);
  tmp1 = tmp1.cwiseProduct(tmp2);
  tmp2 = fft.inv(tmp1);
  return tmp2.head(num_bins_ + 1).real();
}

} // end kde1d::bandwidth

} // end kde1d
