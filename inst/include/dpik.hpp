#pragma once

#include "stats.hpp"
#include <unsupported/Eigen/FFT>
#define _USE_MATH_DEFINES
#include <cmath>

class PluginBandwidthSelector {
public:
    PluginBandwidthSelector(const Eigen::VectorXd& x,
                            const Eigen::VectorXd& weights = Eigen::VectorXd());
    double select_bw(size_t deg);
    double select_nn(size_t deg);

private:
    Eigen::VectorXd linbin(const Eigen::VectorXd& x,
                           const Eigen::VectorXd& weights);
    double scale_est(const Eigen::VectorXd& x);
    Eigen::VectorXd kde_drv(size_t drv);
    void set_bw_for_bkfe(size_t drv);
    double bkfe(size_t drv);
    double ll_ibias2(size_t deg);
    double ll_ibias2_nn(size_t deg);
    double ll_ivar(size_t deg);
    double ll_ivar_nn(size_t deg);

    size_t num_bins_{ 401 };
    double bw_ { NAN };
    double lower_;
    double upper_;
    Eigen::VectorXd weights_;
    Eigen::VectorXd bin_counts_;
    double scale_;
};


PluginBandwidthSelector::PluginBandwidthSelector(const Eigen::VectorXd& x,
                                                 const Eigen::VectorXd& weights)
    : lower_(x.minCoeff())
    , upper_(x.maxCoeff())
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
inline Eigen::VectorXd PluginBandwidthSelector::linbin(const Eigen::VectorXd& x,
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

double PluginBandwidthSelector::scale_est(const Eigen::VectorXd& x)
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
Eigen::VectorXd PluginBandwidthSelector::kde_drv(size_t drv)
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
void PluginBandwidthSelector::set_bw_for_bkfe(size_t drv)
{
    if (drv % 2 != 0) {
        throw std::runtime_error("only even drv allowed.");
    }

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

    bw_ = std::pow(-2 * Kr / (psi * n), 1.0 / (r + 1));
}


// Binned Kernel Functional Estimate
//! @param x vector of bin counts
//! @param drv order of derivative in the density functional
//! @param h kernel bandwidth
//! @param a minimum value of x at which to compute the estimate
//! @param b maximum value of x at which to compute the estimate
//! @return the estimated functional
double PluginBandwidthSelector::bkfe(size_t drv)
{
    return bin_counts_.cwiseProduct(kde_drv(drv)).sum() / bin_counts_.sum();
}

// integrated squared bias without bw and n terms
double PluginBandwidthSelector::ll_ibias2(size_t deg)
{
    Eigen::VectorXd arg;
    // bias expressions from Geenens (JASA, 2014)
    if (deg == 0) {
        set_bw_for_bkfe(4);
        arg = 0.25 * kde_drv(4);
    } else if (deg == 1) {
        set_bw_for_bkfe(4);
        Eigen::VectorXd f0 = kde_drv(0);
        Eigen::VectorXd f1 = kde_drv(1);
        Eigen::VectorXd f2 = kde_drv(2);
        arg = (0.5 * f2 + f1.cwiseAbs2().cwiseQuotient(f0))
            .cwiseAbs2().cwiseQuotient(f0);
    } else if (deg == 2) {
        set_bw_for_bkfe(8);
        Eigen::VectorXd f0 = kde_drv(0);
        Eigen::VectorXd f1 = kde_drv(1);
        Eigen::VectorXd f2 = kde_drv(2);
        Eigen::VectorXd f4 = kde_drv(4);
        arg = f4 - 3 * f2.cwiseAbs2().cwiseQuotient(f0) +
            2 * (f1.array().pow(4) / f0.array().pow(3)).matrix();
        arg = (0.125 * arg).cwiseAbs2().cwiseQuotient(f0);
    } else {
        throw std::runtime_error("deg must be one of {0, 1, 2}.");
    }
    return bin_counts_.cwiseProduct(arg).sum() / bin_counts_.sum();
}

// integrated squared bias without bw and n terms
double PluginBandwidthSelector::ll_ibias2_nn(size_t deg)
{
    // Follows from
    //      d_k(x) ~ (k / n) / (2 f(x)),
    // where d_k(x) is the distance to the k-nearest neighbor and (k / n) the
    // nearest neighbor fraction.
    Eigen::VectorXd arg;
    if (deg == 0) {
        set_bw_for_bkfe(4);
        Eigen::VectorXd f0 = kde_drv(0);
        Eigen::VectorXd f2 = kde_drv(2);
        arg = (0.5 * 0.25 * f2.cwiseQuotient(f0.cwiseAbs2()))
            .cwiseAbs2().cwiseQuotient(f0);
    } else if (deg == 1) {
        set_bw_for_bkfe(4);
        Eigen::VectorXd f0 = kde_drv(0);
        Eigen::VectorXd f1 = kde_drv(1);
        Eigen::VectorXd f2 = kde_drv(2);
        arg = (0.5 * 0.25 * f2 + f1.cwiseAbs2().cwiseQuotient(f0))
            .cwiseQuotient(f0.cwiseAbs2())
            .cwiseAbs2().cwiseQuotient(f0);
    } else if (deg == 2) {
        set_bw_for_bkfe(8);
        Eigen::VectorXd f0 = kde_drv(0);
        Eigen::VectorXd f1 = kde_drv(1);
        Eigen::VectorXd f2 = kde_drv(2);
        Eigen::VectorXd f4 = kde_drv(4);
        arg = f4 - 3 * f2.cwiseAbs2().cwiseQuotient(f0) +
            2 * (f1.array().pow(4) / f0.array().pow(3)).matrix();
        arg = (0.125 * 0.25 * arg).
            cwiseQuotient(f0.cwiseAbs2()).
            cwiseAbs2().cwiseQuotient(f0);
    } else {
        throw std::runtime_error("deg must be one of {0, 1, 2}.");
    }
    return bin_counts_.cwiseProduct(arg).sum() / bin_counts_.sum();
}

// integrated variance without bw and n terms
double PluginBandwidthSelector::ll_ivar(size_t deg)
{
    // variance expressions from Geenens (JASA, 2014)
    if (deg > 2)
        throw std::runtime_error("deg must be one of {0, 1, 2}.");
    return (deg < 2 ? 1.0 : 27.0 / 16.0) * 0.5 / std::sqrt(M_PI);
}

// integrated variance for nn without bw and n terms
double PluginBandwidthSelector::ll_ivar_nn(size_t deg)
{
    // variance expressions from Geenens (JASA, 2014)
    if (deg > 2)
        throw std::runtime_error("deg must be one of {0, 1, 2}.");
    set_bw_for_bkfe(0);
    return (deg < 2 ? 1.0 : 27.0 / 16.0) * bkfe(0) / std::sqrt(M_PI);
}

// Bandwidth for Kernel Density Estimation
//! @param x vector of observations
//! @param weights vector of weights for each observation (can be empty).
//! @param grid_size number of equally-spaced points over which binning is
//! performed to obtain kernel functional approximation
//! @return the selected bandwidth
inline double PluginBandwidthSelector::select_bw(size_t deg)
{
    // effective sample size
    double n = std::pow(weights_.sum(), 2) / weights_.cwiseAbs2().sum();
    double bw;
    int bwpow = (deg < 2 ? 4 : 8);
    try {
        double ibias2 = ll_ibias2(deg);
        double ivar   = ll_ivar(deg);
        bw = std::pow(ivar / (bwpow * n * ibias2), 1.0 / (bwpow + 1));
    } catch (...) {
        bw = 4.0 * 1.06 * scale_ * std::pow(n, -1.0 / (bwpow + 1));
    }

    return bw;
}

//! Bandwidth for Nearest Neighbor Kernel Density Estimation
//! @param x vector of observations
//! @param weights vector of weights for each observation (can be empty).
//! @param grid_size number of equally-spaced points over which binning is
//! performed to obtain kernel functional approximation
//! @return the selected bandwidth
inline double PluginBandwidthSelector::select_nn(size_t deg)
{
    // effective sample size
    double n = std::pow(weights_.sum(), 2) / weights_.cwiseAbs2().sum();
    double nn = 0.5;
    int nnpow = (deg < 2 ? 4 : 8);
    // try {
        double ibias2 = ll_ibias2_nn(deg);
        double ivar   = ll_ivar_nn(deg);
        nn = std::pow(ivar / (nnpow * n * ibias2), 1.0 / (nnpow + 1));
    // } catch (...) {
    // }

    return std::min(nn, 1.0);
}
