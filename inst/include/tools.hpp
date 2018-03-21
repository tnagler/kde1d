#pragma once

#include <Eigen/Dense>

namespace tools {

//! remove rows of a matrix which contain nan values
//! @param x the matrix.
//! @return a new matrix without the rows containing nan values
inline Eigen::MatrixXd nan_omit(const Eigen::MatrixXd &x)
{
    // find rows with nans
    Eigen::Matrix<bool, 1, Eigen::Dynamic>
    nans = x.array().isNaN().matrix().rowwise().any();

    // if there is no nan, just return x
    if (!nans.array().any()) {
        return x;
    }

    // copy data to not modify input
    Eigen::MatrixXd out = x;
    size_t last = x.rows() - 1;
    for (size_t i = 0; i < last + 1;) {
        // put nan rows at the end
        if (nans(i)) {
            out.row(i).swap(out.row(last));
            nans.segment<1>(i).swap(nans.segment<1>(last));
            --last;
        } else {
            ++i;
        }
    }
    out.conservativeResize(last + 1, out.cols());

    return out;
}

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
//! @param n_iter the number of iterations for the bisection (defaults to 35,
//! guaranteeing an accuracy of 0.5^35 ~= 6e-11).
//!
//! @return \f$ f^{-1}(x) \f$.
inline Eigen::VectorXd invert_f(const Eigen::VectorXd &x,
                                std::function<
                                    Eigen::VectorXd(const Eigen::VectorXd &)
                                    > f,
                                    const double lb,
                                    const double ub,
                                    int n_iter
) {
    Eigen::VectorXd xl = Eigen::VectorXd::Constant(x.size(), lb);
    Eigen::VectorXd xh = Eigen::VectorXd::Constant(x.size(), ub);
    Eigen::VectorXd x_tmp = x;
    for (
            int iter = 0;
            iter<n_iter;
            ++iter) {
        x_tmp = (xh + xl) / 2.0;
        Eigen::VectorXd fm = f(x_tmp) - x;
        xl = (fm.array() < 0).select(x_tmp, xl);
        xh = (fm.array() < 0).select(xh, x_tmp);
    }

    return
        x_tmp;
}

}

