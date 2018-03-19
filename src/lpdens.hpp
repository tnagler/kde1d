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
        auto points = Eigen::VectorXd::LinSpaced(grid_size, x_min, x_max);
        auto vals = eval_kde1d(points, x, bw);
        grid_ = InterpolationGrid1d(points, vals, 3);
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

        return invert_f(x, f, 0.0, 1.0, 20);
    }

private:
    InterpolationGrid1d grid_;

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

    Eigen::VectorXd eval_kde1d(const Eigen::VectorXd& x_ev,
                               const Eigen::VectorXd& x,
                               double bw)
    {
        auto f = [&x, &bw, this] (double xx) {
            return this->kern_gauss((x.array() - xx) / bw).mean() / bw;
        };

        return x_ev.unaryExpr(f);
    }
};