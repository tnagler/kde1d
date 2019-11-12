#pragma once

#include <Eigen/Dense>

namespace kde1d {

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
inline Eigen::VectorXd invert_f(
  const Eigen::VectorXd &x,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f,
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

//! finds the index, where the minimum in a vector occurs.
//! @param x the vector.
inline size_t find_min_index(const Eigen::VectorXd& x)
{
  return std::min_element(x.data(), x.data() + x.size()) - x.data();
}

//! remove rows of a matrix which contain nan values or have zero weight
//! @param x the matrix.
//! @param a vector of weights that is either empty or whose size is equal to
//!   the number of columns of x.
inline void remove_nans(Eigen::VectorXd& x, Eigen::VectorXd& weights)
{
  if ((weights.size() > 0) & (weights.size() != x.rows()))
    throw std::runtime_error("sizes of x and weights don't match.");

  // if an entry is nan or weight is zero, move it to the end
  size_t last = x.size() - 1;
  for (size_t i = 0; i < last + 1; i++) {
    bool is_nan = std::isnan(x(i));
    if (weights.size() > 0) {
      is_nan = is_nan | std::isnan(weights(i));
      is_nan = is_nan | (weights(i) == 0.0);
    }
    if (is_nan) {
      if (weights.size() > 0)
        std::swap(weights(i), weights(last));
      std::swap(x(i--), x(last--));
    }
  }

  // remove nan rows
  x.conservativeResize(last + 1);
  if (weights.size() > 0)
    weights.conservativeResize(last + 1);
}

} // end kde1d tools

} // end kde1d
