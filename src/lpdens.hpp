#pragma once

#include "interpolation.hpp"
#include "stats.hpp"
#include <functional>

class LPDens1d {
public:
    LPDens1d() {}

    LPDens1d(Eigen::VectorXd x, double bw, double xmin, double xmax) :
        bw_(bw),
        xmin_(xmin),
        xmax_(xmax)
    {
        loglik_ = 0;
        edf_ = 0;

        // construct equally spaced grid on original domain
        grid_points_ = construct_grid_points(x);

        // transform in case of boundary correction
        grid_points_ = boundary_transform(grid_points_);
        x = boundary_transform(x);

        // fit model and evaluate in transformed domain
        Eigen::MatrixXd fitted = fit_kde1d(grid_points_, x, bw);

        // back-transform grid to original domain
        grid_points_ = boundary_transform(grid_points_, true);

        // correct estimated density for transformation
        values_ = boundary_correct(grid_points_, fitted.col(0));

        // move boundary points to xmin/xmax
        grid_points_ = finalize_grid(grid_points_);

        // construct interpolation grid
        // (3 iterations for normalization to a proper density)
        grid_ = InterpolationGrid1d(grid_points_, values_, 3);

        // store normalized values
        values_ = grid_.get_values();

        // calculate log-likelihood of final estimate
        loglik_ = grid_.interpolate(x).array().log().sum();

        // calculate effective degrees of freedom
        InterpolationGrid1d infl_grid(grid_points_, fitted.col(1), 0);
        edf_ = infl_grid.interpolate(x).sum();
    }

    Eigen::VectorXd d(const Eigen::VectorXd& x)
    {
        return grid_.interpolate(x);
    }

    Eigen::VectorXd p(const Eigen::VectorXd& x)
    {
        return grid_.integrate(x);
    }

    Eigen::VectorXd q(const Eigen::VectorXd& x)
    {
        auto f = [this] (const Eigen::VectorXd& xx) {
            return grid_.integrate(xx);
        };

        return invert_f(x,
                        f,
                        grid_.get_grid_points().minCoeff(),
                        grid_.get_grid_points().maxCoeff(),
                        20);
    }

    Eigen::VectorXd get_values() const {return values_;}
    Eigen::VectorXd get_grid_points() const {return grid_points_;}
    double get_bw() const {return bw_;}
    double get_xmin() const {return xmin_;}
    double get_xmax() const {return xmax_;}
    double get_edf() const {return edf_;}
    double get_loglik() const {return loglik_;}

private:
    InterpolationGrid1d grid_;
    Eigen::VectorXd grid_points_;
    Eigen::VectorXd values_;
    double bw_;
    double xmin_;
    double xmax_;
    double loglik_;
    double edf_;

    Eigen::VectorXd kern_gauss(const Eigen::VectorXd& x)
    {
        auto f = [] (double xx) {
            if (std::fabs(xx) > 5.0)
                return 0.0;
            double val = stats::dnorm(Eigen::VectorXd::Constant(1, xx))(0);
            // correct for truncation
            return val / 0.999999426697;
        };
        return x.unaryExpr(f);
    }

    Eigen::MatrixXd fit_kde1d(const Eigen::VectorXd& x_ev,
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

    Eigen::VectorXd boundary_transform(const Eigen::VectorXd& x,
                                       bool inverse = false)
    {
        Eigen::VectorXd x_new = x;
        if (!inverse) {
            if (!std::isnan(xmin_) & !std::isnan(xmax_)) {  // two boundaries
                x_new = (x.array() - xmin_ + 5e-3) / (xmax_ - xmin_ + 1e-2);
                x_new = stats::qnorm(x_new);
            } else if (!std::isnan(xmin_)) {                // left boundary
                x_new = (1e-3 + x.array() - xmin_).log();
            } else if (!std::isnan(xmax_)) {                // right boundary
                x_new = (1e-3 + xmax_ - x.array()).log();
            }
        } else {
            if (!std::isnan(xmin_) & !std::isnan(xmax_)) {  // two boundaries
                x_new = stats::pnorm(x).array() + xmin_ - 5e-3;
                x_new *=  (xmax_ - xmin_ + 1e-2);
            } else if (!std::isnan(xmin_)) {                // left boundary
                x_new = x.array().exp() + xmin_ - 1e-3;
            } else if (!std::isnan(xmax_)) {                // right boundary
                x_new = -x.array().exp() - xmax_ - 1e-3;
            }
        }

        return x_new;
    }

    Eigen::VectorXd boundary_correct(const Eigen::VectorXd& x,
                                     const Eigen::VectorXd& fhat)
    {
        Eigen::VectorXd corr_term(fhat.size());
        if (!std::isnan(xmin_) & !std::isnan(xmax_)) {  // two boundaries
            corr_term = (x.array() - xmin_ + 5e-3) / (xmax_ - xmin_ + 1e-2);
            corr_term = stats::dnorm(stats::qnorm(corr_term));
            corr_term /= (xmax_ - xmin_ + 1e-2);
            corr_term = 1 / corr_term.array().max(1e-4);
        } else if (!std::isnan(xmin_)) {                // left boundary
            corr_term = 1 / (1e-3 + x.array() - xmin_);
        } else if (!std::isnan(xmax_)) {                // right boundary
            corr_term = 1 / (1e-3 + xmax_ - x.array());
        } else {
            corr_term.fill(1.0);
        }

        return fhat.array() * corr_term.array();
    }

    Eigen::VectorXd construct_grid_points(const Eigen::VectorXd& x)
    {
        double x_min = x.minCoeff();
        double x_max = x.maxCoeff();
        double range = x_max - x_min;

        size_t grid_size = 200;
        Eigen::VectorXd lowr_ext, uppr_ext, grid_points(grid_size);

        // no left boundary -> add a few points to the left
        if (std::isnan(xmin_)) {
            lowr_ext = Eigen::VectorXd::LinSpaced(5,
                                                  x_min - range,
                                                  x_min - 0.1 * range);
            grid_size -= 5;
        } else {
            lowr_ext = Eigen::VectorXd();
        }

        // no right boundary -> add a few points to the right
        if (std::isnan(xmax_)) {
            uppr_ext = Eigen::VectorXd::LinSpaced(5,
                                                  x_max + 0.1 * range,
                                                  x_max + range);
            grid_size -= 5;
        } else {
            uppr_ext = Eigen::VectorXd();
        }

        grid_points <<
            lowr_ext,
            Eigen::VectorXd::LinSpaced(grid_size, x_min, x_max),
            uppr_ext;

        return grid_points;
    }

    Eigen::VectorXd finalize_grid(Eigen::VectorXd& grid)
    {
        if (std::isnan(xmin_))
            grid(0) = xmin_;
        if (std::isnan(xmax_))
            grid(grid.size()) = xmax_;
        return grid;
    }
};