#pragma once

#include <Eigen/Dense>

namespace tools {

//! applies a function to each non-NaN value, otherwise returns NaN
//! @param x function argument.
//! @param func function to be applied.
template<typename T>
Eigen::MatrixXd unaryExpr_or_nan(const Eigen::MatrixXd &x, const T& func)
{
    return x.unaryExpr([&func](double y) {
        if (std::isnan(y)) {
            return std::numeric_limits<double>::quiet_NaN();
        } else {
            return func(y);
        }
    });
}

//! computes the inverse \f$ f^{-1} \f$ of a function \f$ f \f$ by the
//! bisection method.
//!
//! @param x evaluation points.
//! @param f the function to invert.
//! @param lb lower bound.
//! @param ub upper bound.
//! @param n_iter the number of iterations for the bisection.
//!
//! @return \f$ f^{-1}(x) \f$.
inline Eigen::VectorXd invert_f(const Eigen::VectorXd &x,
                                std::function<
                                    Eigen::VectorXd(const Eigen::VectorXd &)
                                > f,
                                const double lb,
                                const double ub,
                                int n_iter)
{
    Eigen::VectorXd xl = Eigen::VectorXd::Constant(x.size(), lb);
    Eigen::VectorXd xh = Eigen::VectorXd::Constant(x.size(), ub);
    Eigen::VectorXd x_tmp = x;
    for (int iter = 0; iter < n_iter; ++iter) {
        x_tmp = (xh + xl) / 2.0;
        Eigen::VectorXd fm = f(x_tmp) - x;
        xl = (fm.array() < 0).select(x_tmp, xl);
        xh = (fm.array() < 0).select(xh, x_tmp);
    }

    return x_tmp;
}

}
