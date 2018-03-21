#pragma once

#include "interpolation.hpp"
#include "stats.hpp"
#include "tools.hpp"
#include <functional>

//! Local-polynomial density estimation in 1-d.
//! (only local constant/ classical kde so far)
class LPDens1d {
public:
    // constructors
    LPDens1d() {}
    LPDens1d(Eigen::VectorXd x, double bw, double xmin, double xmax);

    // getters
    Eigen::VectorXd get_values() const {return grid_.get_values();}
    Eigen::VectorXd get_grid_points() const {return grid_.get_grid_points();}
    double get_bw() const {return bw_;}
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
    double loglik_;
    double edf_;

    // private methods
    Eigen::VectorXd kern_gauss(const Eigen::VectorXd& x);
    Eigen::MatrixXd fit_kde1d(const Eigen::VectorXd& x_ev,
                              const Eigen::VectorXd& x,
                              double bw);
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
inline LPDens1d::LPDens1d(Eigen::VectorXd x,
                          double bw,
                          double xmin,
                          double xmax) :
    bw_(bw),
    xmin_(xmin),
    xmax_(xmax)
{
    // construct equally spaced grid on original domain
    Eigen::VectorXd grid_points = construct_grid_points(x);

    // transform in case of boundary correction
    grid_points = boundary_transform(grid_points);
    x = boundary_transform(x);

    // fit model and evaluate in transformed domain
    Eigen::MatrixXd fitted = fit_kde1d(grid_points, x, bw);

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

        // otherwise calculate normal pdf
        double val = stats::dnorm(Eigen::VectorXd::Constant(1, xx))(0);

        return val / 0.999999426697;  // correct for truncation
    };
    return x.unaryExpr(f);
}

//! (analytically) evaluates the kernel density estimate and its influence
//! function on a user-supplied grid.
//! @param x_ev evaluation points.
//! @param x observations.
//! @param bw the bandwidth.
//! @return a two-column matrix containing the density estimate in the first
//!   and the influence function in the second column.
inline Eigen::MatrixXd LPDens1d::fit_kde1d(const Eigen::VectorXd& x_ev,
                                           const Eigen::VectorXd& x,
                                           double bw)
{
    Eigen::MatrixXd out(x_ev.size(), 2);

    // density estimate
    auto fhat = [&x, &bw, this] (double xx) {
        return this->kern_gauss((x.array() - xx) / bw).mean() / bw;
    };
    out.col(0) = x_ev.unaryExpr(fhat);

    // influence function estimate
    double contrib = kern_gauss(Eigen::VectorXd::Zero(1))(0) / bw;
    contrib /= static_cast<double>(x.size());
    out.col(1) = contrib / out.col(0).array();

    return out;
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
            x_new = (x.array() - xmin_ + 5e-3) / (xmax_ - xmin_ + 1e-2);
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
            x_new = stats::pnorm(x).array() + xmin_ - 5e-3;
            x_new *=  (xmax_ - xmin_ + 1e-2);
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
        corr_term = (x.array() - xmin_ + 5e-3) / (xmax_ - xmin_ + 1e-2);
        corr_term = stats::dnorm(stats::qnorm(corr_term));
        corr_term /= (xmax_ - xmin_ + 1e-2);
        corr_term = 1.0 / corr_term.array().max(1e-4);
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
