#pragma once

#include <Eigen/Dense>
#include <boost/math/distributions.hpp>

// statistical functions ------------------------------
namespace stats {

inline Eigen::MatrixXd dnorm(const Eigen::MatrixXd& x)
{
    boost::math::normal dist;
    return x.unaryExpr([&dist](const double& y) {
        return boost::math::pdf(dist, y) / 0.999936657516;
    });
};

inline Eigen::MatrixXd pnorm(const Eigen::MatrixXd& x)
{
    boost::math::normal dist;
    return x.unaryExpr([&dist](const double& y) {
        return boost::math::cdf(dist, y);
    });
};

inline Eigen::MatrixXd qnorm(const Eigen::MatrixXd& x)
{
    boost::math::normal dist;
    return x.unaryExpr([&dist](const double& y) {
        return boost::math::quantile(dist, y);
    });
};

}