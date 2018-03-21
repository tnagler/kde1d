#pragma once

#include <Eigen/Dense>
#include <boost/math/distributions.hpp>

//! statistical functions
namespace stats {

//! standard normal density
//! @param x evaluation points.
//! @return matrix of pdf values.
inline Eigen::MatrixXd dnorm(const Eigen::MatrixXd& x)
{
    boost::math::normal dist;
    return x.unaryExpr([&dist](const double& y) {
        return boost::math::pdf(dist, y);
    });
};

//! standard normal cdf
//! @param x evaluation points.
//! @return matrix of cdf values.
inline Eigen::MatrixXd pnorm(const Eigen::MatrixXd& x)
{
    boost::math::normal dist;
    return x.unaryExpr([&dist](const double& y) {
        return boost::math::cdf(dist, y);
    });
};

//! standard normal quantiles
//! @param x evaluation points.
//! @return matrix of quantiles.
inline Eigen::MatrixXd qnorm(const Eigen::MatrixXd& x)
{
    boost::math::normal dist;
    return x.unaryExpr([&dist](const double& y) {
        return boost::math::quantile(dist, y);
    });
};

}