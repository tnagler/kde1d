#pragma once

#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false
#include <Eigen/Dense>
#include <boost/math/distributions.hpp>
#include <algorithm>
#include <vector>

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

//! empirical quantiles
//! @param x data.
//! @param q evaluation points.
//! @return vector of quantiles.
inline Eigen::VectorXd quantile(const Eigen::VectorXd& x,
                                const Eigen::VectorXd& q)
{
    double n = static_cast<double>(x.size());
    size_t m = q.size();
    Eigen::VectorXd res(m);

    // map to std::vector and sort
    std::vector<double> x2(x.data(), x.data() + x.size());
    std::sort(x2.begin(), x2.end());

    // linear interpolation (quantile of type 7 in R)
    for (size_t i = 0; i < m; ++i) {
        size_t k = std::floor(n * q(i));
        double p = (static_cast<double>(k) - 1.0) / (n - 1.0);
        res(i) = x2[k - 1] + (x2[k] - x2[k - 1]) * (q(i) - p) * (n - 1);
    }
    return res;
}

//! empirical quantiles
//! @param x data.
//! @param q evaluation points.
//! @param w vector of weights.
//! @return vector of quantiles.
inline Eigen::VectorXd quantile(const Eigen::VectorXd& x,
                                const Eigen::VectorXd& q,
                                const Eigen::VectorXd& w)
{
    if (w.size() != x.size())
        throw std::runtime_error("x and w must have the same size");
    double n = static_cast<double>(x.size());
    size_t m = q.size();
    Eigen::VectorXd res(m);

    // map to std::vector and sort
    std::vector<size_t> ind(n);
    for (size_t i = 0; i < n; ++i)
        ind[i] = i;
    std::sort(ind.begin(), ind.end(),
              [&x] (size_t i, size_t j) { return x(i) < x(j); });

    auto x2 = x;
    auto wcum = w;
    auto wrank = Eigen::VectorXd::Constant(n, 0.0);
    double wacc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x2(i) = x(ind[i]);
        wcum(i) = wacc;
        wacc += w(ind[i]);
    }

    double wsum = w.sum() - w(ind[n - 1]);;
    for (size_t j = 0; j < m; ++j) {
        size_t i = 1;
        while (wcum(i) < q(j) * wsum)
            i++;
        res(j) = x2(i - 1) + (x2(i) - x2(i - 1)) *
            (q(j) - wcum(i - 1) / wsum) / (wcum(i) - wcum(i - 1));
    }

    return res;
}

}
