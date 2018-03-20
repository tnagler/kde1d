#pragma once

#include "interpolation.hpp"
#include <functional>

class LPDens1d {
public:
    LPDens1d() {}

    LPDens1d(const Eigen::VectorXd& x, size_t grid_size, double bw)
    {
        double x_min = x.minCoeff();
        double x_max = x.maxCoeff();
        double range = x_max - x_min;
        x_min = x_min - 0.5 * range;
        x_max = x_max + 0.5 * range;
        grid_points_ = Eigen::VectorXd::LinSpaced(grid_size, x_min, x_max);
        Eigen::MatrixXd fitted = fit_kde1d(grid_points_, x, bw);
        values_ = fitted.col(0);
        grid_ = InterpolationGrid1d(grid_points_, values_, 3);
        values_ = grid_.get_values();
        loglik_ = grid_.interpolate(x).array().log().sum();

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
    double get_edf() const {return edf_;}
    double get_loglik() const {return loglik_;}

private:
    InterpolationGrid1d grid_;
    Eigen::VectorXd grid_points_;
    Eigen::VectorXd values_;
    double loglik_;
    double edf_;

    Eigen::VectorXd kern_gauss(const Eigen::VectorXd& x)
    {
        auto f = [] (double xx) {
            if (std::fabs(xx) > 5.0)
                return 0.0;
            // normalize by 0.9999994267 because of truncation
            return exp(- 0.5 * std::pow(xx, 2)) / (sqrt(2 * M_PI)) / 0.9999994267;
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
        out.col(1) = contrib / out.col(0).array();

        return out;
    }
};