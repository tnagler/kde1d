#pragma once

#include "stats.hpp"
#include <unsupported/Eigen/FFT>
#define _USE_MATH_DEFINES
#include <cmath>

class BinnedKDE {
public:
    BinnedKDE(const Eigen::VectorXd& x,
              double bw,
              const Eigen::VectorXd& weights = Eigen::VectorXd(),
              size_t num_bins = 401);

    void set_bw(double bw) { bw_ = bw; }
    Eigen::VectorXd kde_drv(size_t drv);
    double bkfe(size_t drv);
    double dpik(size_t deg);

private:
    Eigen::VectorXd linbin(const Eigen::VectorXd& x,
                           const Eigen::VectorXd& weights);
    double scale_est(const Eigen::VectorXd& x);
    double select_bw_for_bkfe(size_t drv);

    double lower_;
    double upper_;
    size_t num_bins_;
    double bw_;
    Eigen::VectorXd weights_;
    Eigen::VectorXd bin_counts_;
    double scale_;
};


BinnedKDE::BinnedKDE(const Eigen::VectorXd& x,
                     double bw,
                     const Eigen::VectorXd& weights,
                     size_t num_bins)
    : lower_(x.minCoeff())
    , upper_(x.maxCoeff())
    , num_bins_(num_bins)
    , bw_(bw)
    , weights_(weights)
{
    if (weights.size() > 0 && (weights.size() != x.size()))
        throw std::runtime_error("x and weights must have the same size");
    if (weights.size() == 0) {
        weights_ = Eigen::VectorXd::Ones(x.size());
    } else {
        weights_ = weights_ * x.size() / weights_.sum();
    }

    bin_counts_ = linbin(x, weights_);
    scale_ = scale_est(x);
}

// Obtains bin counts for univariate data via the linear binning strategy.
//! @param x vector of observations
//! @param weights vector of weights for each observation.
//! @return bin counts
inline Eigen::VectorXd BinnedKDE::linbin(const Eigen::VectorXd& x,
                                         const Eigen::VectorXd& weights)
{
    Eigen::VectorXd gcnts = Eigen::VectorXd::Zero(num_bins_);
    double rem, lxi, delta;

    delta = (upper_ - lower_) / (num_bins_ - 1.0);
    for (size_t i = 0; i < x.size(); ++i) {
        lxi = (x(i) - lower_) / delta + 1.0;
        size_t li = static_cast<size_t>(lxi);
        rem = lxi - li;
        if (li >= 1 && li < num_bins_) {
            gcnts(li - 1) += (1 - rem) * weights(i);
            gcnts(li) += rem * weights(i);
        }
    }

    return gcnts;
}

double BinnedKDE::scale_est(const Eigen::VectorXd& x)
{
    double m_x = x.cwiseProduct(weights_).mean();
    Eigen::VectorXd sx = (x - Eigen::VectorXd::Constant(x.size(), m_x));
    double sd_x = std::sqrt(
        sx.cwiseAbs2().cwiseProduct(weights_).sum()/(x.size() - 1));
    Eigen::VectorXd q_x(2);
    q_x << 0.25, 0.75;
    q_x = stats::quantile(x, q_x, weights_);
    double scale = std::min((q_x(1) - q_x(0))/1.349, sd_x);
    if (scale == 0) {
        scale = (sd_x > 0) ? sd_x : 1.0;
    }
    return scale;
}


//! Binned kernel density derivative estimate
//! @param x vector of bin counts
//! @param drv order of derivative
//! @return estimated derivative evaluated at the grid points
Eigen::VectorXd BinnedKDE::kde_drv(size_t drv)
{
    double delta = (upper_ - lower_) / (num_bins_ - 1.0);
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

//! optimal bandwidths for kernel functionals (see Wand and Jones' book, 3.5)
//! only works for even drv
double BinnedKDE::select_bw_for_bkfe(size_t drv)
{
    if (drv % 2 != 0) {
        throw std::runtime_error("only even drv allowed.");
    }
    double bw_old = bw_;

    // effective sample size
    double n = std::pow(weights_.sum(), 2) / weights_.cwiseAbs2().sum();

    // start with normal reference rule (eq 3.7)
    int r = drv + 4;
    double psi =((r/2) % 2 == 0) ? 1 : -1;
    psi *= std::tgamma(r + 1);
    psi /= std::pow(2 * scale_, r + 1) * std::tgamma(r/2 + 1) * std::sqrt(M_PI);
    double Kr = stats::dnorm_drv(Eigen::VectorXd::Zero(1), r - 2)(0);
    bw_ = std::pow(-2 * Kr / (psi * n), 1.0 / (r + 1));

    // now use plug in to select the actual bandwidth (eq. 3.6)
    r -= 2;
    psi = bkfe(drv + 2);
    Kr = stats::dnorm_drv(Eigen::VectorXd::Zero(1), r - 2)(0);

    bw_ = bw_old;
    return std::pow(-2 * Kr / (psi * n), 1.0 / (r + 1));
}


// Binned Kernel Functional Estimate
//! @param x vector of bin counts
//! @param drv order of derivative in the density functional
//! @param h kernel bandwidth
//! @param a minimum value of x at which to compute the estimate
//! @param b maximum value of x at which to compute the estimate
//! @return the estimated functional
double BinnedKDE::bkfe(size_t drv)
{
    return bin_counts_.cwiseProduct(kde_drv(drv)).sum() / bin_counts_.sum();
}

//! Binned Kernel Functional Estimate for nearest neighbor bandwidths
//! @param x vector of observations
//! @param h kernel bandwidth
//! @param a minimum value of x at which to compute the estimate
//! @param b maximum value of x at which to compute the estimate
//! @return the estimated functional
double bkfe_nn(const Eigen::VectorXd& x, double h, double a, double b)
{
    // Eigen::VectorXd drv2 = bkdrv(x, 2, h, a, b).cwiseAbs2();
    // Eigen::VectorXd pdf3 = bkdrv(x, 0, h, a, b).array().pow(3);
    // Eigen::VectorXd arg = drv2.array() / pdf3.array();
    // arg = arg.unaryExpr([] (const double& xx) {
    //     return std::isnan(xx) ? 0.0 : xx;
    // });
    return 1.0;// x.cwiseProduct(arg).sum() / x.sum();
}


// Bandwidth for Kernel Density Estimation
//! @param x vector of observations
//! @param weights vector of weights for each observation (can be empty).
//! @param grid_size number of equally-spaced points over which binning is
//! performed to obtain kernel functional approximation
//! @return the selected bandwidth
inline double BinnedKDE::dpik(size_t deg)
{
    // effective sample size
    double n = std::pow(weights_.sum(), 2) / weights_.cwiseAbs2().sum();
    double bw = 1.0;
    double old_bw = bw_;
    try {
        bw_ = select_bw_for_bkfe(4);
        double psi = bkfe(4);
        double del0 = 1.0 / std::pow(4.0 * M_PI, 1.0 / 10.0);
        bw = del0 * std::pow(1.0 / (psi * n), 1.0 / 5.0);
    } catch (...) {
        bw = 4.0 * 1.06 * scale_ * std::pow(1.0 / n, 1.0 / 5.0);
    }

    bw_ = old_bw;
    return bw;
}

//! Bandwidth for Nearest Neighbor Kernel Density Estimation
//! @param x vector of observations
//! @param weights vector of weights for each observation (can be empty).
//! @param grid_size number of equally-spaced points over which binning is
//! performed to obtain kernel functional approximation
//! @return the selected bandwidth
inline double dpik_nn(const Eigen::VectorXd& x,
                      double bw,
                      Eigen::VectorXd weights = Eigen::VectorXd(),
                      size_t grid_size = 401)
{
    // if (weights.size() > 0 && (weights.size() != x.size()))
    //     throw std::runtime_error("x and weights must have the same size");
    //
    // if (weights.size() == 0) {
    //     weights = Eigen::VectorXd::Constant(x.size(), 1.0);
    // } else {
    //     weights = weights * x.size() / weights.sum();
    // }
    //
    // double n = static_cast<double>(x.size());
    // double a = x.minCoeff();
    // double b = x.maxCoeff();
    //
    // double m_x = x.cwiseProduct(weights).mean();
    // Eigen::VectorXd sx = (x - Eigen::VectorXd::Constant(x.size(), m_x));
    // double sd_x = std::sqrt(sx.cwiseAbs2().cwiseProduct(weights).sum()/(n - 1));
    // Eigen::VectorXd q_x(2);
    // q_x(0) = 0.75;
    // q_x(1) = 0.25;
    // q_x = stats::quantile(x, q_x, weights);
    // double scale = std::min((q_x(0) - q_x(1))/1.349, sd_x);
    // if (scale == 0) {
    //     scale = (sd_x > 0) ? sd_x : 1.0;
    // }
    //
    // sx /= scale;
    // double sa = (a - m_x) / scale;
    // double sb = (b - m_x) / scale;
    // auto x2 = linbin(sx, sa, sb, grid_size, weights);
    //
    // double nn = 0.0;
    // try {
    //     double effn = std::pow(weights.sum(), 2) / weights.cwiseAbs2().sum();
    //     double bfun = bkfe_nn(x2, bw * std::pow(effn, 1 / 5.0 - 1 / 7.0), a, b);
    //     double vfun = bkfe(x2, 0, bw, a, b);
    //
    //     double del0 = 1.0 / std::pow(4.0 * M_PI, 1.0 / 10.0);
    //     nn = scale * del0 * std::pow(vfun / (bfun * effn), 1.0 / 5.0);
    // } catch (...) {}

    return 0.0; //std::min(nn, 1.0);
}
