#pragma once

#include "interpolation.hpp"
#include "stats.hpp"
#include "tools.hpp"
#include <functional>

//! Local-polynomial density estimation in 1-d.
class LPDens1d {
public:
    // constructors
    LPDens1d() {}
    LPDens1d(Eigen::VectorXd x, double bw, double xmin, double xmax, size_t p);

    // getters
    Eigen::VectorXd get_values() const {return grid_.get_values();}
    Eigen::VectorXd get_grid_points() const {return grid_.get_grid_points();}
    double get_bw() const {return bw_;}
    double get_p() const {return deg_;}
    double get_xmin() const {return xmin_;}
    double get_xmax() const {return xmax_;}
    double get_edf() const {return edf_;}
    double get_loglik() const {return loglik_;}

private:
    // data members
    InterpolationGrid1d grid_;
    double bw_;
    double xmin_;
    double xmax_;
    size_t deg_;
    double loglik_;
    double edf_;

    // private methods
    Eigen::VectorXd kern_gauss(const Eigen::VectorXd& x);
    Eigen::MatrixXd fit_lp(const Eigen::VectorXd& x_ev,
                           const Eigen::VectorXd& x);
    double calculate_infl(const size_t& n,
                          const double& f0,
                          const double& b,
                          const double& bw,
                          const double& s);
    Eigen::VectorXd boundary_transform(const Eigen::VectorXd& x,
                                       bool inverse = false);
    Eigen::VectorXd boundary_correct(const Eigen::VectorXd& x,
                                     const Eigen::VectorXd& fhat);
    Eigen::VectorXd construct_grid_points(const Eigen::VectorXd& x);
    Eigen::VectorXd finalize_grid(Eigen::VectorXd& grid_points);
    Eigen::VectorXd without_boundary_ext(const Eigen::VectorXd& grid_points);
};

//! constructor for fitting the density estimate.
//! @param x vector of observations
//! @param bw positive bandwidth parameter
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
//! @param p order of the local polynomial.
inline LPDens1d::LPDens1d(Eigen::VectorXd x,
                          double bw,
                          double xmin,
                          double xmax,
                          size_t deg) :
    bw_(bw),
    xmin_(xmin),
    xmax_(xmax),
    deg_(deg)
{
    // construct equally spaced grid on original domain
    Eigen::VectorXd grid_points = construct_grid_points(x);

    // transform in case of boundary correction
    grid_points = boundary_transform(grid_points);
    x = boundary_transform(x);

    // fit model and evaluate in transformed domain
    Eigen::MatrixXd fitted = fit_lp(grid_points, x);

    // back-transform grid to original domain
    grid_points = boundary_transform(grid_points, true);
    x = boundary_transform(x, true);

    // correct estimated density for transformation
    Eigen::VectorXd values = boundary_correct(grid_points, fitted.col(0));

    // move boundary points to xmin/xmax
    grid_points = finalize_grid(grid_points);

    // construct interpolation grid
    // (3 iterations for normalization to a proper density)
    grid_ = InterpolationGrid1d(grid_points, values, 3);

    // store normalized values
    values = grid_.get_values();

    // calculate log-likelihood of final estimate
    loglik_ = grid_.interpolate(x).array().log().sum();

    // calculate effective degrees of freedom
    InterpolationGrid1d infl_grid(without_boundary_ext(grid_points),
                                  without_boundary_ext(fitted.col(1)),
                                  0);
    edf_ = infl_grid.interpolate(x).sum();
}

//! Gaussian kernel (truncated at +/- 5).
//! @param x vector of evaluation points.
inline Eigen::VectorXd LPDens1d::kern_gauss(const Eigen::VectorXd& x)
{
    auto f = [] (double xx) {
        // truncate at +/- 5
        if (std::fabs(xx) > 5.0)
            return 0.0;
        // otherwise calculate normal pdf (orrect for truncation)
        return  stats::dnorm(Eigen::VectorXd::Constant(1, xx))(0) / 0.999999426;
    };
    return x.unaryExpr(f);
}

//! (analytically) evaluates the kernel density estimate and its influence
//! function on a user-supplied grid.
//! @param x_ev evaluation points.
//! @param x observations.
//! @return a two-column matrix containing the density estimate in the first
//!   and the influence function in the second column.
inline Eigen::MatrixXd LPDens1d::fit_lp(const Eigen::VectorXd& x_ev,
                                        const Eigen::VectorXd& x)
{
    Eigen::MatrixXd res(x_ev.size(), 2);
    size_t n = x.size();

    double f0, f1, b;
    double s = bw_;
    Eigen::VectorXd xx(x.size());
    Eigen::VectorXd xx2(x.size());
    Eigen::VectorXd kernels(x.size());
    for (size_t k = 0; k < x_ev.size(); k++) {
        // classical (local constant) kernel density estimate
        xx = (x.array() - x_ev(k)) / bw_;
        kernels = kern_gauss(xx) / bw_;
        f0 = kernels.mean();
        res(k, 0) = f0;

        if (deg_ > 0) {
            // calculate b for local linear
            xx /= bw_;
            f1 = xx.cwiseProduct(kernels).mean(); // first order derivative
            b = f1 / f0;

            if (deg_ == 2) {
                // more calculations for local quadratic
                xx2 = xx.cwiseProduct(kernels) / (f0 * static_cast<double>(n));
                b *= std::pow(bw_, 2);
                s = 1.0 / (std::pow(bw_, 4) * xx.transpose() * xx2 - std::pow(b, 2));
                res(k, 0) *= bw_ * std::sqrt(s);
            }

            // final estimate
            res(k, 0) *= std::exp(-0.5 * std::pow(b, 2) * s);
            if ((boost::math::isnan)(res(k)) |
                (boost::math::isinf)(res(k))) {
                // inverse operation might go wrong due to rounding when
                // true value is equal or close to zero
                res(k, 0) = 0.0;
            }
        }

        // influence function estimate
        res(k, 1) = calculate_infl(n, f0, b, bw_, s);
    }

    return res;
}

//! calculate influence for data point for density estimate based on
//! quantities pre-computed in `fit_lp()`.
inline double LPDens1d::calculate_infl(const size_t &n,
                                       const double& f0,
                                       const double& b,
                                       const double& bw,
                                       const double& s)
{
    Eigen::MatrixXd M;
    double bw2 = std::pow(bw, 2);
    double b2 = std::pow(b, 2);
    if (deg_ == 0) {
        M = Eigen::MatrixXd::Constant(1, 1, f0);
    } else if (deg_ == 1) {
        M = Eigen::MatrixXd(2, 2);
        M(0, 0) = f0;
        M(0, 1) = bw2 * b * f0;
        M(1, 0) = M(0, 1);
        M(1, 1) = f0 * bw2 + f0 * bw2 * bw2 * b2;
    } else if (deg_ == 2) {
        M = Eigen::MatrixXd(3, 3);
        M(0, 0) = f0;
        M(0, 1) = f0 * b;
        M(1, 0) = M(0, 1);
        M(1, 1) = f0 * bw2 + f0 * b2;
        M(0, 2) = M(2, 2);
        M(2, 0) = M(2, 2);
        M(1, 2) = 0.5 * f0 * (3.0 / s * b + b * b2);
        M(2, 1) = M(1, 2);
        M(2, 2) = 3.0 / std::pow(s, 2) + 6.0 * b2 / std::pow(s, 3);
        M(2, 2) = 0.25 * f0;
        M(2, 2) *= 3.0 / std::pow(s, 2) + 6.0 / s * b2  + b2 * b2;
    }

    double infl = kern_gauss(Eigen::VectorXd::Zero(1))(0) / bw;
    infl *= M.inverse()(0, 0) / static_cast<double>(n);
    return infl;
}


//! transformations for density estimates with bounded support.
//! @param x evaluation points.
//! @param inverse whether the inverse transformation should be applied.
//! @return the transformed evaluation points.
inline Eigen::VectorXd LPDens1d::boundary_transform(const Eigen::VectorXd& x,
                                                    bool inverse)
{
    Eigen::VectorXd x_new = x;
    if (!inverse) {
        if (!std::isnan(xmin_) & !std::isnan(xmax_)) {
            // two boundaries -> probit transform
            x_new = (x.array() - xmin_ + 5e-5) / (xmax_ - xmin_ + 1e-4);
            x_new = stats::qnorm(x_new);
        } else if (!std::isnan(xmin_)) {
            // left boundary -> log transform
            x_new = (1e-3 + x.array() - xmin_).log();
        } else if (!std::isnan(xmax_)) {
            // right boundary -> negative log transform
            x_new = (1e-3 + xmax_ - x.array()).log();
        } else {
            // no boundary -> no transform
        }
    } else {
        if (!std::isnan(xmin_) & !std::isnan(xmax_)) {
            // two boundaries -> probit transform
            x_new = stats::pnorm(x).array() + xmin_ - 5e-5;
            x_new *=  (xmax_ - xmin_ + 1e-4);
        } else if (!std::isnan(xmin_)) {
            // left boundary -> log transform
            x_new = x.array().exp() + xmin_ - 1e-3;
        } else if (!std::isnan(xmax_)) {
            // right boundary -> negative log transform
            x_new = -(x.array().exp() - xmax_ - 1e-3);
        } else {
            // no boundary -> no transform
        }
    }

    return x_new;
}

//! corrects the density estimate for a preceding boundary transformation of
//! the data.
//! @param x evaluation points (in original domain).
//! @param fhat the density estimate evaluated in the transformed domain.
//! @return corrected density estimates at `x`.
inline Eigen::VectorXd LPDens1d::boundary_correct(const Eigen::VectorXd& x,
                                                  const Eigen::VectorXd& fhat)
{
    Eigen::VectorXd corr_term(fhat.size());
    if (!std::isnan(xmin_) & !std::isnan(xmax_)) {
        // two boundaries -> probit transform
        corr_term = (x.array() - xmin_ + 5e-5) / (xmax_ - xmin_ + 1e-4);
        corr_term = stats::dnorm(stats::qnorm(corr_term));
        corr_term /= (xmax_ - xmin_ + 1e-4);
        corr_term = 1.0 / corr_term.array().max(1e-6);
    } else if (!std::isnan(xmin_)) {
        // left boundary -> log transform
        corr_term = 1.0 / (1e-3 + x.array() - xmin_);
    } else if (!std::isnan(xmax_)) {
        // right boundary -> negative log transform
        corr_term = 1.0 / (1e-3 + xmax_ - x.array());
    } else {
        // no boundary -> no transform
        corr_term.fill(1.0);
    }

    return fhat.array() * corr_term.array();
}

//! constructs a grid that is later used for interpolation.
//! @param x vector of observations.
//! @return a grid of size 50.
inline Eigen::VectorXd LPDens1d::construct_grid_points(const Eigen::VectorXd& x)
{
    double x_min = x.minCoeff();
    double x_max = x.maxCoeff();
    double range = x_max - x_min;

    size_t grid_size = 50;
    Eigen::VectorXd lowr_ext, uppr_ext, grid_points(grid_size);

    // no left boundary -> add a few points to the left
    if (std::isnan(xmin_)) {
        lowr_ext = Eigen::VectorXd::LinSpaced(5,
                                              x_min - 0.5 * range,
                                              x_min - 0.05 * range);
        grid_size -= 5;
    } else {
        lowr_ext = Eigen::VectorXd();
    }

    // no right boundary -> add a few points to the right
    if (std::isnan(xmax_)) {
        uppr_ext = Eigen::VectorXd::LinSpaced(5,
                                              x_max + 0.05 * range,
                                              x_max + 0.5 * range);
        grid_size -= 5;
    } else {
        uppr_ext = Eigen::VectorXd();
    }

    // concatenate
    grid_points <<
        lowr_ext,
        Eigen::VectorXd::LinSpaced(grid_size, x_min, x_max),
        uppr_ext;

    return grid_points;
}

//! moves the boundary points of the grid to xmin/xmax (if non-NaN).
//! @param grid_points the grid points.
inline Eigen::VectorXd LPDens1d::finalize_grid(Eigen::VectorXd& grid_points)
{
    double range = grid_points.maxCoeff() - grid_points.minCoeff();
    if (!std::isnan(xmin_))
        grid_points(0) = xmin_;
    if (!std::isnan(xmax_))
        grid_points(grid_points.size() - 1) = xmax_;

    return grid_points;
}

//! removes the boundary extension from the grid_points (see
//! `construct_grid_points`).
//! @param grid_points the grid points.
inline Eigen::VectorXd LPDens1d::without_boundary_ext(
        const Eigen::VectorXd& grid_points)
{
    size_t grid_start = 0;
    size_t grid_size = grid_points.size();
    // (grid extension has length 5)
    if (std::isnan(xmin_)) {
        grid_start += 4;
        grid_size -= 5;
    }
    if (std::isnan(xmax_))
        grid_size -= 5;

    return grid_points.segment(grid_start, grid_size);
}
