#pragma once

#include <Eigen/Dense>
#include <functional>
#include "tools.hpp"

namespace kde1d {

namespace interp {

//! A class for cubic spline interpolation in one dimension
//!
//! The class is used for implementing kernel estimators. It makes storing the
//! observations obsolete and allows for fast numerical integration.
class InterpolationGrid1d
{
public:
  InterpolationGrid1d() {}

  InterpolationGrid1d(const Eigen::VectorXd& grid_points,
                      const Eigen::VectorXd& values,
                      int norm_times);

  void normalize(int times);

  Eigen::VectorXd interpolate(const Eigen::VectorXd& x) const;

  Eigen::VectorXd integrate(const Eigen::VectorXd& u) const;

  Eigen::VectorXd get_values() const {return values_;}
  Eigen::VectorXd get_grid_points() const {return grid_points_;}

private:
  // Utility functions for spline Interpolation
  double cubic_poly(const double& x,
                    const Eigen::VectorXd& a) const;

  double cubic_indef_integral(const double& x,
                              const Eigen::VectorXd& a) const;

  double cubic_integral(const double& lower,
                        const double& upper,
                        const Eigen::VectorXd& a) const;

  Eigen::VectorXd find_coefs(const Eigen::VectorXd& vals,
                             const Eigen::VectorXd& grid) const;

  double interp_on_grid(const double& x,
                        const Eigen::VectorXd& vals,
                        const Eigen::VectorXd& grid) const;

  ptrdiff_t find_cell(const double& x0) const;

  // Utility functions for integration
  double int_on_grid(const double& upr,
                     const Eigen::VectorXd& vals,
                     const Eigen::VectorXd& grid) const;

  Eigen::VectorXd grid_points_;
  Eigen::MatrixXd values_;
};


//! Constructor
//!
//! @param grid_points an ascending sequence of grid points.
//! @param values a vector of values of same length as grid_points.
//! @param norm_times how many times the normalization routine should run.
inline InterpolationGrid1d::InterpolationGrid1d(
  const Eigen::VectorXd& grid_points,
  const Eigen::VectorXd& values,
  int norm_times)
{
  if (grid_points.size() != values.size())
    throw std::runtime_error("grid_points and values must be of equal length");

  grid_points_ = grid_points;
  values_ = values;
  this->normalize(norm_times);
}

//! renormalizes the estimate to integrate to one
//!
//! @param times how many times the normalization routine should run.
inline void InterpolationGrid1d::normalize(int times)
{
  double x_max = grid_points_(grid_points_.size() - 1);
  double int_max;
  for (int k = 0; k < times; ++k) {
    int_max = int_on_grid(x_max, values_, grid_points_);
    values_ /= int_max;
  }
}

inline ptrdiff_t InterpolationGrid1d::find_cell(const double& x0) const
{
  ptrdiff_t low = 0, high = grid_points_.size() - 1;
  ptrdiff_t mid;
  while (low < high - 1) {
    mid = low + (high - low) / 2;
    if (x0 < grid_points_(mid))
      high = mid;
    else
      low = mid;
  }

  return low;
}

//! Interpolation
//! @param x vector of evaluation points.
inline Eigen::VectorXd InterpolationGrid1d::interpolate(
    const Eigen::VectorXd& x) const
{
  Eigen::VectorXd tmpgrid(4), tmpvals(4);
  ptrdiff_t m = grid_points_.size();

  auto interpolate_one = [&] (const double& xx) {
    ptrdiff_t i0, i3;
    ptrdiff_t i = find_cell(xx);
    i0 = std::max(i - 1, static_cast<ptrdiff_t>(0));
    i3 = std::min(i + 2, m - 1);
    tmpgrid(0) = this->grid_points_(i0);
    tmpgrid(1) = this->grid_points_(i);
    tmpgrid(2) = this->grid_points_(i + 1);
    tmpgrid(3) = this->grid_points_(i3);
    tmpvals(0) = this->values_(i0);
    tmpvals(1) = this->values_(i);
    tmpvals(2) = this->values_(i + 1);
    tmpvals(3) = this->values_(i3);
    return this->interp_on_grid(xx, tmpvals, tmpgrid);
  };

  return tools::unaryExpr_or_nan(x, interpolate_one);
}

//! Integration along the grid
//!
//! @param x a vector  of evaluation points
inline Eigen::VectorXd InterpolationGrid1d::integrate(const Eigen::VectorXd& x)
  const
{
  Eigen::VectorXd res(x.size());
  auto ord = tools::get_order(x);

  // temporaries for the loop
  Eigen::VectorXd tmpvals(4), tmpgrid(4), tmpa(4);
  double new_int, new_upr, grid_eps, cum_int = 0.0;
  int k = 0, m = grid_points_.size();

  for (size_t i = 0; i < x.size(); ++i) {
    double upr = x(ord(i));
    if (upr <= grid_points_(0)) {
      res(ord(i)) = cum_int;
      continue;
    }
    // go up the grid and integrate
    while (k < m - 1) {
      // halt loop if x(k) is below integral boundary
      if (upr < grid_points_(k + 1))
        break;
      // select length 4 subvectors and calculate spline coefficients
      tmpvals(0) = values_(std::max(k - 1, 0));
      tmpvals(1) = values_(k);
      tmpvals(2) = values_(k + 1);
      tmpvals(3) = values_(std::min(k + 2, m - 1));

      tmpgrid(0) = grid_points_(std::max(k - 1, 0));
      tmpgrid(1) = grid_points_(k);
      tmpgrid(2) = grid_points_(k + 1);
      tmpgrid(3) = grid_points_(std::min(k + 2, m - 1));

      tmpa = find_coefs(tmpvals, tmpgrid);

      // integrate over full cell
      grid_eps = (grid_points_(k + 1) - grid_points_(k));
      cum_int += cubic_integral(0.0, 1.0, tmpa) * grid_eps;
      k++;
    }

    // integrate over partial cell
    upr = (upr - grid_points_(k)) / grid_eps;
    new_int = cubic_integral(0.0, std::min(upr, 1.0), tmpa) * grid_eps;
    res(ord(i)) = cum_int + new_int;
  }

  // TODO: integrate until end to normalize all values
  return res;
}

// ---------------- Utility functions for spline interpolation ----------------

//! Evaluate a cubic polynomial
//!
//! @param x evaluation point.
//! @param a polynomial coefficients
inline double InterpolationGrid1d::cubic_poly(const double& x,
                                              const Eigen::VectorXd& a) const
{
  double x2 = x * x;
  double x3 = x2 * x;
  return a(0) + a(1) * x + a(2) * x2 + a(3) * x3;
}

//! Indefinite integral of a cubic polynomial
//!
//! @param x evaluation point.
//! @param a polynomial coefficients.
inline double InterpolationGrid1d::cubic_indef_integral(
        const double& x, const Eigen::VectorXd& a) const
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
inline double InterpolationGrid1d::cubic_integral(const double& lower,
                                                  const double& upper,
                                                  const Eigen::VectorXd& a)
  const
{
  return cubic_indef_integral(upper, a) - cubic_indef_integral(lower, a);
}

//! Calculate coefficients for cubic intrpolation spline
//!
//! @param vals length 4 vector of function values.
//! @param grid length 4 vector of grid points.
inline Eigen::VectorXd InterpolationGrid1d::find_coefs(
  const Eigen::VectorXd& vals, const Eigen::VectorXd& grid) const
{
  double dt0 = grid(1) - grid(0);
  double dt1 = grid(2) - grid(1);
  double dt2 = grid(3) - grid(2);

  // compute tangents when parameterized in (t1,t2)
  // for smooth extrapolation, derivative is set to zero at boundary
  double dx1 = 0.0, dx2 = 0.0;
  if (dt0 > 0) {
    dx1 = (vals(1) - vals(0)) / dt0;
    dx1 -= (vals(2) - vals(0)) / (dt0 + dt1);
    dx1 += (vals(2) - vals(1)) / dt1;
  }
  if (dt2 > 0) {
    dx2 = (vals(2) - vals(1)) / dt1;
    dx2 -= (vals(3) - vals(1)) / (dt1 + dt2);
    dx2 += (vals(3) - vals(2)) / dt2;
  }

  // rescale tangents for parametrization in (0,1)
  dx1 *= dt1;
  dx2 *= dt1;

  // ensure positivity (Schmidt and Hess, DOI:10.1007/bf01934097)
  dx1 = std::max(dx1, -3 * vals(1));
  dx2 = std::min(dx2, 3 * vals(2));

  // compute coefficents
  Eigen::VectorXd a(4);
  a(0) = vals(1);
  a(1) = dx1;
  a(2) = -3 * (vals(1) - vals(2)) - 2 * dx1 - dx2;
  a(3) = 2 * (vals(1) - vals(2)) + dx1 + dx2;

  return a;
}

//! Interpolate on 4 points
//!
//! @param x evaluation point.
//! @param vals length 4 vector of function values.
//! @param grid length 4 vector of grid points.
inline double InterpolationGrid1d::interp_on_grid(const double& x,
                                                  const Eigen::VectorXd& vals,
                                                  const Eigen::VectorXd& grid)
  const
{
  double xev = (x - grid(1)) / (grid(2) - grid(1));
  // use Gaussian tail for extrapolation
  if (xev <= 0) {
    return vals(1) * std::exp(-0.5 * xev * xev);
  } else if (xev >= 1) {
    return vals(2) * std::exp(-0.5 * xev * xev);
  }

  Eigen::VectorXd a = find_coefs(vals, grid);
  return cubic_poly(xev, a);
}


// ---------------- Utility functions for integration ----------------


//! Integrate a spline interpolant
//!
//! @param upr upper limit of integration (lower is 0).
//! @param vals vector of values to be interpolated and integrated.
//! @param grid vector of grid points on which vals has been computed.
//!
//! @return Integral of interpolation spline defined by (vals, grid).
inline double InterpolationGrid1d::int_on_grid(const double& upr,
                                               const Eigen::VectorXd& vals,
                                               const Eigen::VectorXd& grid)
  const
{
  ptrdiff_t m = grid.size();
  Eigen::VectorXd tmpvals(4), tmpgrid(4), tmpa(4);
  double uprnew, newint;

  double tmpint = 0.0;

  if (upr > grid(0)) {
      // go up the grid and integrate
    for (ptrdiff_t k = 0; k < m - 1; ++k) {
        // stop loop if fully integrated
      if (upr < grid(k))
        break;

      // select length 4 subvectors and calculate spline coefficients
      tmpvals(0) = vals(std::max(k - 1, static_cast<ptrdiff_t>(0)));
      tmpvals(1) = vals(k);
      tmpvals(2) = vals(k + 1);
      tmpvals(3) = vals(std::min(k + 2, m - 1));

      tmpgrid(0) = grid(std::max(k - 1, static_cast<ptrdiff_t>(0)));
      tmpgrid(1) = grid(k);
      tmpgrid(2) = grid(k + 1);
      tmpgrid(3) = grid(std::min(k + 2, m - 1));

      tmpa = find_coefs(tmpvals, tmpgrid);

      // don't integrate over full cell if upr is in interior
      uprnew = (upr - grid(k)) / (grid(k + 1) - grid(k));
      newint = cubic_integral(0.0, std::fmin(1.0, uprnew), tmpa);
      tmpint += std::fmax(newint, 0.0) * (grid(k + 1) - grid(k));
    }
  }

  return tmpint;
}

} // end kde1d::interp

} // end kde1d