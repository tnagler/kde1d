#pragma once

#include <Eigen/Dense>

namespace kde1d {

namespace tools {

//! applies a function to each non-NaN value, otherwise returns NaN
//! @param x function argument.
//! @param func function to be applied.
template<typename T>
Eigen::MatrixXd
unaryExpr_or_nan(const Eigen::MatrixXd& x, const T& func)
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
inline Eigen::VectorXd
invert_f(const Eigen::VectorXd& x,
         std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f,
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

//! remove rows of a matrix which contain nan values or have zero weight
//! @param x the matrix.
//! @param a vector of weights that is either empty or whose size is equal to
//!   the number of columns of x.
inline void
remove_nans(Eigen::VectorXd& x, Eigen::VectorXd& weights)
{
  if ((weights.size() > 0) && (weights.size() != x.rows()))
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

inline Eigen::VectorXi
get_order(const Eigen::VectorXd& x)
{
  Eigen::VectorXi order(x.size());
  for (long i = 0; i < x.size(); ++i)
    order(i) = static_cast<int>(i);
  std::stable_sort(order.data(),
                   order.data() + order.size(),
                   [&](const size_t& a, const size_t& b) {
                     return std::isnan(x[a]) || (x[a] < x[b]);
                   });
  return order;
}

//! Computes bin counts for univariate data via the linear binning strategy.
//! @param x vector of observations
//! @param weights vector of weights for each observation.
inline Eigen::VectorXd
linbin(const Eigen::VectorXd& x,
       double lower,
       double upper,
       size_t num_bins,
       const Eigen::VectorXd& weights)
{
  Eigen::VectorXd gcnts = Eigen::VectorXd::Zero(num_bins + 1);
  double delta = (upper - lower) / static_cast<double>(num_bins);
  double rem, lxi;
  size_t li;
  for (long i = 0; i < x.size(); ++i) {
    lxi = (x(i) - lower) / delta;
    li = static_cast<size_t>(lxi);
    rem = lxi - static_cast<double>(li);
    if (li < num_bins) {
      gcnts(li) += (1 - rem) * weights(i);
      gcnts(li + 1) += rem * weights(i);
    }
  }

  return gcnts;
}

} // end kde1d tools

} // end kde1d
