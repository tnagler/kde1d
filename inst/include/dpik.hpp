#pragma once

#include "stats.hpp"
#include <unsupported/Eigen/FFT>
#define _USE_MATH_DEFINES
#include <cmath>

// Obtains bin counts for univariate data via the linear binning strategy.
//! @param x vector of observations
//! @param a lower bound
//! @param b upper bound
//! @param m number of bins
//! @return bin counts
inline Eigen::VectorXd linbin(const Eigen::VectorXd& x,
                              double a, double b, size_t m)
{
    Eigen::VectorXd gcnts = Eigen::VectorXd::Zero(m);
    double rem, lxi, delta;

    delta = (b - a) / (static_cast<double>(m) - 1.0);
    for (size_t i = 0; i < x.size(); ++i) {
        lxi = (x(i) - a) / delta + 1.0;
        size_t li = static_cast<size_t>(lxi);
        rem = lxi - li;
        if (li >= 1 && li < m) {
            gcnts(li - 1) += 1 - rem;
            gcnts(li) += rem;
        }
    }
    return(gcnts);
}

// Binned Kernel Functional Estimate
//! @param x vector of observations
//! @param drv order of derivative in the density functional
//! @param h kernel bandwidth
//! @param a minimum value of x at which to compute the estimate
//! @param b maximum value of x at which to compute the estimate
//! @return the estimated functional
double bkfe(const Eigen::VectorXd& x,
            size_t drv, double h, double a, double b)
{
    size_t m = x.size();
    double delta = (b - a) / (static_cast<double>(m) - 1.0);
    double tau = 4.0 + static_cast<double>(drv);
    size_t L = std::min(static_cast<size_t>(std::floor(tau * h / delta)), m);

    double tmp_dbl = static_cast<double>(L) * delta / h;
    Eigen::VectorXd arg = Eigen::VectorXd::LinSpaced(L + 1, 0.0, tmp_dbl);

    Eigen::VectorXd hmold0 = Eigen::VectorXd::Ones(L + 1);
    Eigen::VectorXd hmold1 = arg;
    Eigen::VectorXd hmnew = Eigen::VectorXd::Ones(L + 1);
    for (size_t i = 2; i <= drv; i++) {
        hmnew = arg.cwiseProduct(hmold1) - hmold0 * static_cast<double>(i - 1);
        hmold0 = hmold1;
        hmold1 = hmnew;
    }

    tmp_dbl = std::pow(h, static_cast<double>(drv + 1));
    arg = stats::dnorm(arg) / tmp_dbl;
    arg = arg.cwiseProduct(hmnew);

    tmp_dbl = static_cast<double>(m + L + 1);
    tmp_dbl = std::pow(2, std::ceil((std::log(tmp_dbl)/std::log(2))));
    size_t P = static_cast<size_t>(tmp_dbl);

    Eigen::VectorXd arg2 = Eigen::VectorXd::Zero(P);
    arg2.block(0, 0, L + 1, 1) = arg;
    arg2.block(P - L, 0, L, 1) = arg.block(1, 0, L, 1).colwise().reverse();

    Eigen::VectorXd x2 = Eigen::VectorXd::Zero(P);
    x2.block(0, 0, m, 1) = x;

    Eigen::FFT<double> fft;
    Eigen::VectorXcd tmp1 = fft.fwd(arg2);
    Eigen::VectorXcd tmp2 = fft.fwd(x2);
    tmp1 = tmp1.cwiseProduct(tmp2);
    tmp2 = fft.inv(tmp1);
    Eigen::VectorXd x3 = tmp2.real().block(0, 0, m, 1);
    return(x.cwiseProduct(x3).sum()/std::pow(x.sum(), 2));
}

// Bandwidth for Kernel Density Estimation
//! @param x vector of observations
//! @param grid_size number of equally-spaced points over which binning is
//! performed to obtain kernel functional approximation
//! @return the selected bandwidth
inline double dpik(const Eigen::VectorXd& x, size_t grid_size = 401) {

    double n = static_cast<double>(x.size());

    double a = x.minCoeff();
    double b = x.maxCoeff();

    double m_x = x.mean();
    Eigen::VectorXd sx = (x - Eigen::VectorXd::Constant(x.size(), m_x));
    double sd_x = std::sqrt(sx.cwiseAbs2().sum()/(n - 1));
    Eigen::VectorXd q_x(2);
    q_x(0) = 0.75;
    q_x(1) = 0.25;
    q_x = stats::quantile(x, q_x);
    double scale = std::min((q_x(0) - q_x(1))/1.349, sd_x);

    double bw = 0.0;
    try {
        sx /= scale;
        double sa = (a - m_x) / scale;
        double sb = (b - m_x) / scale;
        auto x2 = linbin(sx, sa, sb, grid_size);

        double alpha = std::pow(std::pow(2.0, 11.0 / 2.0)/(7.0 * n), 1.0 / 9.0);
        double psi6hat = bkfe(x2, 6, alpha, sa, sb);
        alpha = std::pow(-3.0 * std::sqrt(2.0 / M_PI)/(psi6hat * n), 1.0 / 7.0);
        double psi4hat = bkfe(x2, 4, alpha, sa, sb);

        double del0 = 1.0 / std::pow(4.0 * M_PI, 1.0 / 10.0);
        bw = scale * del0 * std::pow(1.0 / (psi4hat * n), 1.0 / 5.0);
    } catch (...) {
        bw = 4.0 * 1.06 *scale * std::pow(1.0 / n, 1.0 / 5.0);
    }

    return(bw);
}
