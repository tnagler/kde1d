#pragma once

#include "tools.hpp"
#include <Eigen/Dense>

namespace kde1d {

namespace interp {

//! A class for cubic spline interpolation in one dimension
//!
//! The class is used for implementing kernel estimators. It makes storing the
//! observations obsolete and allows for fast numerical integration.
class InterpolationGrid
{
public:
  InterpolationGrid() {}

  InterpolationGrid(const Eigen::VectorXd& grid_points,
                    const Eigen::VectorXd& values,
                    int norm_times);

  void normalize(int times);

  Eigen::VectorXd interpolate(const Eigen::VectorXd& x) const;

  Eigen::VectorXd integrate(const Eigen::VectorXd& u,
                            bool normalize = false) const;

  Eigen::VectorXd get_values() const { return values_; }
  Eigen::VectorXd get_grid_points() const { return grid_points_; }
  double get_grid_max() const { return grid_points_[grid_points_.size() - 1]; }
  double get_grid_min() const { return grid_points_[0]; }

private:
  // Utility functions for spline Interpolation
  double cubic_poly(const double& x, const Eigen::VectorXd& a) const;
  double cubic_indef_integral(const double& x, const Eigen::VectorXd& a) const;
  double cubic_integral(const double& lower,
                        const double& upper,
                        const Eigen::VectorXd& a) const;
  size_t find_cell(const double& x0) const;
  Eigen::VectorXd find_cell_coefs(const size_t& k) const;

  Eigen::VectorXd grid_points_;
  Eigen::VectorXd values_;
};

//! Constructor
//!
//! @param grid_points an ascending sequence of grid points.
//! @param values a vector of values of same length as grid_points.
//! @param norm_times how many times the normalization routine should run.
inline InterpolationGrid::InterpolationGrid(const Eigen::VectorXd& grid_points,
                                            const Eigen::VectorXd& values,
                                            int norm_times)
{
  if (grid_points.size() != values.size())
    throw std::invalid_argument(
      "grid_points and values must be of equal length");

  grid_points_ = grid_points;
  values_ = values;
  this->normalize(norm_times);
}

//! renormalizes the estimate to integrate to one
//!
//! @param times how many times the normalization routine should run.
inline void
InterpolationGrid::normalize(int times)
{
  double x_max = grid_points_(grid_points_.size() - 1);
  double int_max;
  for (int k = 0; k < times; ++k) {
    int_max = this->integrate(Eigen::VectorXd::Constant(1, x_max))(0);
    values_ /= int_max;
  }
}

//! Interpolation
//! @param x vector of evaluation points.
inline Eigen::VectorXd
InterpolationGrid::interpolate(const Eigen::VectorXd& x) const
{
  Eigen::VectorXd tmp_coefs(4);
  auto interpolate_one = [&](const double& xx) {
    size_t k = find_cell(xx);
    double xev =
      (xx - grid_points_(k)) / (grid_points_(k + 1) - grid_points_(k));

    // use Gaussian tail for extrapolation
    if (xev <= 0) {
      return values_(k) * std::exp(-0.5 * xev * xev);
    } else if (xev >= 1) {
      return values_(k + 1) * std::exp(-0.5 * xev * xev);
    }

    return cubic_poly(xev, find_cell_coefs(k));
  };

  return tools::unaryExpr_or_nan(x, interpolate_one);
}

//! Integration along the grid
//!
//! @param x a vector  of evaluation points
//! @param normalize whether to normalize the integral to a maximum value of 1.
inline Eigen::VectorXd
InterpolationGrid::integrate(const Eigen::VectorXd& x, bool normalize) const
{
  Eigen::VectorXd res(x.size());
  auto ord = tools::get_order(x);

  // temporaries for the loop
  Eigen::VectorXd tmp_coefs(4);
  double new_int, tmp_eps, cum_int = 0.0;
  size_t k = 0, m = grid_points_.size();
  tmp_coefs = find_cell_coefs(0);
  tmp_eps = (grid_points_(1) - grid_points_(0));

  for (long i = 0; i < x.size(); ++i) {
    double upr = x(ord(i));

    if (std::isnan(upr)) {
      res(ord(i)) = upr;
      continue;
    }
    if (upr <= grid_points_(0)) {
      res(ord(i)) = 0.0;
      continue;
    }

    // go up the grid and integrate
    while (k < m - 1) {
      // halt loop if integration limit is in kth cell
      if (upr < grid_points_(k + 1))
        break;
      // integrate over full cell
      tmp_coefs = find_cell_coefs(k);
      tmp_eps = (grid_points_(k + 1) - grid_points_(k));
      cum_int += cubic_integral(0.0, 1.0, tmp_coefs) * tmp_eps;
      k++;
    }

    // integrate over partial cell
    if (upr < grid_points_(m - 1)) { // only if still in interior
      tmp_coefs = find_cell_coefs(k);
      tmp_eps = (grid_points_(k + 1) - grid_points_(k));
      upr = (upr - grid_points_(k)) / tmp_eps;
      new_int = cubic_integral(0.0, upr, tmp_coefs) * tmp_eps;
    } else {
      new_int = 0.0;
    }

    res(ord(i)) = cum_int + new_int;
  }

  if (!normalize)
    return res;

  // integrate until end
  while (k < m - 1) {
    tmp_coefs = find_cell_coefs(k);
    tmp_eps = (grid_points_(k + 1) - grid_points_(k));
    cum_int += cubic_integral(0.0, 1.0, tmp_coefs) * tmp_eps;
    k++;
  }
  return res / cum_int;
}

// ---------------- Utility functions for spline interpolation ----------------

//! Evaluate a cubic polynomial
//!
//! @param x evaluation point.
//! @param a polynomial coefficients
inline double
InterpolationGrid::cubic_poly(const double& x, const Eigen::VectorXd& a) const
{
  double x2 = x * x;
  double x3 = x2 * x;
  return a(0) + a(1) * x + a(2) * x2 + a(3) * x3;
}

//! Indefinite integral of a cubic polynomial
//!
//! @param x evaluation point.
//! @param a polynomial coefficients.
inline double
InterpolationGrid::cubic_indef_integral(const double& x,
                                        const Eigen::VectorXd& a) const
{
  double x2 = x * x;
  double x3 = x2 * x;
  double x4 = x3 * x;
  return a(0) * x + a(1) / 2.0 * x2 + a(2) / 3.0 * x3 + a(3) / 4.0 * x4;
}

//! Definite integral of a cubic polynomial
//!
//! @param lower lower limit of the integral.
//! @param upper upper limit of the integral.
//! @param a polynomial coefficients.
inline double
InterpolationGrid::cubic_integral(const double& lower,
                                  const double& upper,
                                  const Eigen::VectorXd& a) const
{
  return cubic_indef_integral(upper, a) - cubic_indef_integral(lower, a);
}

inline size_t
InterpolationGrid::find_cell(const double& x0) const
{
  size_t low = 0, high = grid_points_.size() - 1;
  size_t mid;
  while (low < high - 1) {
    mid = low + (high - low) / 2;
    if (x0 < grid_points_(mid))
      high = mid;
    else
      low = mid;
  }

  return low;
}

//! Calculate coefficients for cubic intrpolation spline
//!
//! @param k the cell index.
inline Eigen::VectorXd
InterpolationGrid::find_cell_coefs(const size_t& k) const
{
  // indices for cell and neighboring grid points
  long int k0 =
    std::max(static_cast<long int>(k) - 1, static_cast<long int>(0));
  long k2 = k + 1;
  long k3 = std::min(static_cast<long int>(k + 2),
                     static_cast<long int>(grid_points_.size() - 1));

  double dt0 = grid_points_(k) - grid_points_(k0);
  double dt1 = grid_points_(k2) - grid_points_(k);
  double dt2 = grid_points_(k3) - grid_points_(k2);

  // compute tangents when parameterized in (t1,t2)
  // for smooth extrapolation, derivative is set to zero at boundary
  double dx1 = 0.0, dx2 = 0.0;
  if (dt0 > 0) {
    dx1 = (values_(k) - values_(k0)) / dt0;
    dx1 -= (values_(k2) - values_(k0)) / (dt0 + dt1);
    dx1 += (values_(k2) - values_(k)) / dt1;
  }
  if (dt2 > 0) {
    dx2 = (values_(k2) - values_(k)) / dt1;
    dx2 -= (values_(k3) - values_(k)) / (dt1 + dt2);
    dx2 += (values_(k3) - values_(k2)) / dt2;
  }

  // rescale tangents for parametrization in (0,1)
  dx1 *= dt1;
  dx2 *= dt1;

  // ensure positivity (Schmidt and Hess, DOI:10.1007/bf01934097)
  dx1 = std::max(dx1, -3 * values_(k));
  dx2 = std::min(dx2, 3 * values_(k2));

  // compute coefficents
  Eigen::VectorXd a(4);
  a(0) = values_(k);
  a(1) = dx1;
  a(2) = -3 * (values_(k) - values_(k2)) - 2 * dx1 - dx2;
  a(3) = 2 * (values_(k) - values_(k2)) + dx1 + dx2;

  return a;
}

} // end kde1d::interp

} // end kde1d