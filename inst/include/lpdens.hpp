#pragma once

#include "interpolation.hpp"
#include "stats.hpp"
#include "tools.hpp"
#include <cmath>
#include <functional>

//! Local-polynomial density estimation in 1-d.
class LPDens1d {
public:
  // constructors
  LPDens1d() {}
  LPDens1d(Eigen::VectorXd x, double bw, double nn, double xmin, double xmax,
           size_t p, const Eigen::VectorXd &weights = Eigen::VectorXd());

  // getters
  Eigen::VectorXd get_values() const { return grid_.get_values(); }
  Eigen::VectorXd get_grid_points() const { return grid_.get_grid_points(); }
  double local_bw(const double &x_ev, const Eigen::VectorXd &x,
                  const double &bw, const double &nn);
  double get_bw() const { return bw_; }
  double get_p() const { return deg_; }
  double get_xmin() const { return xmin_; }
  double get_xmax() const { return xmax_; }
  double get_edf() const { return edf_; }
  double get_loglik() const { return loglik_; }

private:
  // data members
  InterpolationGrid1d grid_;
  double bw_;
  double nn_;
  double xmin_;
  double xmax_;
  size_t deg_;
  double loglik_;
  double edf_;

  // private methods
  Eigen::VectorXd kern_gauss(const Eigen::VectorXd &x);
  Eigen::MatrixXd fit_lp(const Eigen::VectorXd &x_ev, const Eigen::VectorXd &x,
                         const Eigen::VectorXd &weights,
                         const double tol = 1e-6);
  double calculate_infl(const size_t &n, const double &f0, const double &b,
                        const double &bw, const double &s,
                        const double &weight);
  Eigen::VectorXd boundary_transform(const Eigen::VectorXd &x,
                                     bool inverse = false);
  Eigen::VectorXd boundary_correct(const Eigen::VectorXd &x,
                                   const Eigen::VectorXd &fhat);
  Eigen::VectorXd construct_grid_points(const Eigen::VectorXd &x,
                                        const Eigen::VectorXd &weights);
  Eigen::VectorXd finalize_grid(Eigen::VectorXd &grid_points);
  Eigen::VectorXd without_boundary_ext(const Eigen::VectorXd &grid_points);
};

//! constructor for fitting the density estimate.
//! @param x vector of observations
//! @param bw positive bandwidth parameter (fixed component).
//! @param nn nearest neighbor component (between 0 and 1)
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
//! @param p order of the local polynomial.
//! @param weights vector of weights for each observation (can be empty).
inline LPDens1d::LPDens1d(Eigen::VectorXd x, double bw, double nn, double xmin,
                          double xmax, size_t deg,
                          const Eigen::VectorXd &weights)
    : bw_(bw), nn_(nn), xmin_(xmin), xmax_(xmax), deg_(deg) {
  if (weights.size() > 0 && (weights.size() != x.size()))
    throw std::runtime_error("x and weights must have the same size");

  // construct grid on original domain
  Eigen::VectorXd grid_points = construct_grid_points(x, weights);

  // transform in case of boundary correction
  grid_points = boundary_transform(grid_points);
  x = boundary_transform(x);

  // fit model and evaluate in transformed domain
  Eigen::MatrixXd fitted = fit_lp(grid_points, x, weights);

  // back-transform grid to original domain
  grid_points = boundary_transform(grid_points, true);
  x = boundary_transform(x, true);

  // correct estimated density for transformation
  Eigen::VectorXd values = boundary_correct(grid_points, fitted.col(0));

  // move boundary points to xmin/xmax
  grid_points = finalize_grid(grid_points);

  // construct interpolation grid
  // (3 iterations for normalization to a proper density)
  grid_ = InterpolationGrid1d(grid_points, values, 3);

  // store normalized values
  values = grid_.get_values();

  // calculate log-likelihood of final estimate
  loglik_ = grid_.interpolate(x).array().log().sum();

  // calculate effective degrees of freedom
  InterpolationGrid1d infl_grid(without_boundary_ext(grid_points),
                                without_boundary_ext(fitted.col(1)), 0);
  edf_ = infl_grid.interpolate(x).sum();
}

//! Gaussian kernel (truncated at +/- 5).
//! @param x vector of evaluation points.
inline Eigen::VectorXd LPDens1d::kern_gauss(const Eigen::VectorXd &x) {
  auto f = [](double xx) {
    // truncate at +/- 5
    if (std::fabs(xx) > 5.0)
      return 0.0;
    // otherwise calculate normal pdf (orrect for truncation)
    return stats::dnorm(Eigen::VectorXd::Constant(1, xx))(0) / 0.999999426;
  };
  return x.unaryExpr(f);
}

//! computes the local bandwidth for a data point; only relevant when a
//! nearest-neighbor component is specified.
//! @param x_ev the evaluation point.
//! @param x the data.
//! @param bw the fixed bandwidth component.
//! @param nn the nearest-neighbor bandwidth component.
inline double LPDens1d::local_bw(const double &x_ev, const Eigen::VectorXd &x,
                                 const double &bw, const double &nn) {
  if (nn > 0.0) {
    // calculate distances from observations to evaluation point
    Eigen::VectorXd dists = (x.array() - x_ev).abs();

    // calculate index of neighbor such that alpha * n observations are used
    size_t k = std::lround(nn * static_cast<double>(x.size()));
    std::nth_element(dists.data(), dists.data() + k,
                     dists.data() + dists.size());
    return std::max(dists(k), bw);
  } else {
    return bw;
  }
}

//! (analytically) evaluates the kernel density estimate and its influence
//! function on a user-supplied grid.
//! @param x_ev evaluation points.
//! @param x observations.
//! @param weights vector of weights for each observation (can be empty).
//! @return a two-column matrix containing the density estimate in the first
//!   and the influence function in the second column.
inline Eigen::MatrixXd LPDens1d::fit_lp(const Eigen::VectorXd &x_ev,
                                        const Eigen::VectorXd &x,
                                        const Eigen::VectorXd &weights,
                                        const double tol) {
  Eigen::MatrixXd res(x_ev.size(), 2);
  size_t n = x.size();

  double f0, f1, b;
  Eigen::VectorXd xx(x.size());
  Eigen::VectorXd xx2(x.size());
  Eigen::VectorXd kernels(x.size());
  double scale = std::abs(xx.maxCoeff());
  for (size_t k = 0; k < x_ev.size(); k++) {
    double bw = local_bw(x_ev(k), x, bw_, nn_);
    double s = bw;
    // classical (local constant) kernel density estimate
    xx = (x.array() - x_ev(k)) / bw;
    kernels = kern_gauss(xx) / bw;
    if (weights.size() > 0)
      kernels = kernels.cwiseProduct(weights);
    f0 = kernels.mean();
    res(k, 0) = f0;

    if (deg_ > 0) {
      // calculate b for local linear
      xx /= bw;
      f1 = xx.cwiseProduct(kernels).mean(); // first order derivative
      b = f1 / f0;

      if (deg_ == 2) {
        // more calculations for local quadratic
        xx2 = xx.cwiseProduct(kernels);

        if (std::abs(xx2.maxCoeff()) > tol * scale) {
          xx2 /= f0 * static_cast<double>(n);
          b *= std::pow(bw, 2);
          s = 1.0 / (std::pow(bw, 4) * xx.transpose() * xx2 - std::pow(b, 2));
          res(k, 0) *= bw * std::sqrt(s);
        }
      }

      // final estimate
      res(k, 0) *= std::exp(-0.5 * std::pow(b, 2) * s);
      if ((boost::math::isnan)(res(k)) | (boost::math::isinf)(res(k))) {
        // inverse operation might go wrong due to rounding when
        // true value is equal or close to zero
        res(k, 0) = 0.0;
      }
    }

    // influence function estimate
    if (weights.size() > 0) {
      res(k, 1) = calculate_infl(n, f0, b, bw, s, weights(k));
    } else {
      res(k, 1) = calculate_infl(n, f0, b, bw, s, 1.0);
    }
  }

  return res;
}

//! calculate influence for data point for density estimate based on
//! quantities pre-computed in `fit_lp()`.
inline double LPDens1d::calculate_infl(const size_t &n, const double &f0,
                                       const double &b, const double &bw,
                                       const double &s, const double &weight) {
  Eigen::MatrixXd M;
  double bw2 = std::pow(bw, 2);
  double b2 = std::pow(b, 2);
  if (deg_ == 0) {
    M = Eigen::MatrixXd::Constant(1, 1, f0);
  } else if (deg_ == 1) {
    M = Eigen::MatrixXd(2, 2);
    M(0, 0) = f0;
    M(0, 1) = bw2 * b * f0;
    M(1, 0) = M(0, 1);
    M(1, 1) = f0 * bw2 + f0 * bw2 * bw2 * b2;
  } else if (deg_ == 2) {
    M = Eigen::MatrixXd(3, 3);
    M(0, 0) = f0;
    M(0, 1) = f0 * b;
    M(1, 0) = M(0, 1);
    M(1, 1) = f0 * bw2 + f0 * b2;
    M(1, 2) = 0.5 * f0 * (3.0 / s * b + b * b2);
    M(2, 1) = M(1, 2);
    M(2, 2) = 0.25 * f0;
    M(2, 2) *= 3.0 / std::pow(s, 2) + 6.0 / s * b2 + b2 * b2;
    M(0, 2) = M(2, 2);
    M(2, 0) = M(2, 2);
  }

  double infl = kern_gauss(Eigen::VectorXd::Zero(1))(0) / bw;
  infl *= M.inverse()(0, 0) * weight / static_cast<double>(n);
  return infl;
}

//! transformations for density estimates with bounded support.
//! @param x evaluation points.
//! @param inverse whether the inverse transformation should be applied.
//! @return the transformed evaluation points.
inline Eigen::VectorXd LPDens1d::boundary_transform(const Eigen::VectorXd &x,
                                                    bool inverse) {
  Eigen::VectorXd x_new = x;
  if (!inverse) {
    if (!std::isnan(xmin_) & !std::isnan(xmax_)) {
      // two boundaries -> probit transform
      x_new = (x.array() - xmin_ + 5e-5) / (xmax_ - xmin_ + 1e-4);
      x_new = stats::qnorm(x_new);
    } else if (!std::isnan(xmin_)) {
      // left boundary -> log transform
      x_new = (1e-3 + x.array() - xmin_).log();
    } else if (!std::isnan(xmax_)) {
      // right boundary -> negative log transform
      x_new = (1e-3 + xmax_ - x.array()).log();
    } else {
      // no boundary -> no transform
    }
  } else {
    if (!std::isnan(xmin_) & !std::isnan(xmax_)) {
      // two boundaries -> probit transform
      x_new = stats::pnorm(x).array() + xmin_ - 5e-5;
      x_new *= (xmax_ - xmin_ + 1e-4);
    } else if (!std::isnan(xmin_)) {
      // left boundary -> log transform
      x_new = x.array().exp() + xmin_ - 1e-3;
    } else if (!std::isnan(xmax_)) {
      // right boundary -> negative log transform
      x_new = -(x.array().exp() - xmax_ - 1e-3);
    } else {
      // no boundary -> no transform
    }
  }

  return x_new;
}

//! corrects the density estimate for a preceding boundary transformation of
//! the data.
//! @param x evaluation points (in original domain).
//! @param fhat the density estimate evaluated in the transformed domain.
//! @return corrected density estimates at `x`.
inline Eigen::VectorXd LPDens1d::boundary_correct(const Eigen::VectorXd &x,
                                                  const Eigen::VectorXd &fhat) {
  Eigen::VectorXd corr_term(fhat.size());
  if (!std::isnan(xmin_) & !std::isnan(xmax_)) {
    // two boundaries -> probit transform
    corr_term = (x.array() - xmin_ + 5e-5) / (xmax_ - xmin_ + 1e-4);
    corr_term = stats::dnorm(stats::qnorm(corr_term));
    corr_term /= (xmax_ - xmin_ + 1e-4);
    corr_term = 1.0 / corr_term.array().max(1e-6);
  } else if (!std::isnan(xmin_)) {
    // left boundary -> log transform
    corr_term = 1.0 / (1e-3 + x.array() - xmin_);
  } else if (!std::isnan(xmax_)) {
    // right boundary -> negative log transform
    corr_term = 1.0 / (1e-3 + xmax_ - x.array());
  } else {
    // no boundary -> no transform
    corr_term.fill(1.0);
  }

  return fhat.array() * corr_term.array();
}

//! constructs a grid that is later used for interpolation.
//! @param x vector of observations.
//! @return a grid of size 50.
inline Eigen::VectorXd
LPDens1d::construct_grid_points(const Eigen::VectorXd &x,
                                const Eigen::VectorXd &weights) {
  // set up grid
  size_t grid_size = 50;
  Eigen::VectorXd grid_points(grid_size), inner_grid;

  // determine "inner" grid by sample quantiles
  // (need to leave room for boundary extensions)
  if (std::isnan(xmin_))
    grid_size -= 2;
  if (std::isnan(xmax_))
    grid_size -= 2;
  inner_grid =
      stats::quantile(x, Eigen::VectorXd::LinSpaced(grid_size, 0, 1), weights);

  // extend grid where there's no boundary
  double range = inner_grid.maxCoeff() - inner_grid.minCoeff();
  Eigen::VectorXd lowr_ext, uppr_ext;
  if (std::isnan(xmin_)) {
    // no left boundary -> add a few points to the left
    lowr_ext = Eigen::VectorXd(2);
    double step = inner_grid[1] - inner_grid[0];
    lowr_ext[1] = inner_grid[0] - step;
    lowr_ext[0] = lowr_ext[1] - std::max(0.4 * range, step);
  }
  if (std::isnan(xmax_)) {
    // no right boundary -> add a few points to the right
    uppr_ext = Eigen::VectorXd(2);
    double step = inner_grid[grid_size - 1] - inner_grid[grid_size - 2];
    uppr_ext[0] = inner_grid[grid_size - 1] + step;
    uppr_ext[1] = uppr_ext[0] + std::max(0.4 * range, step);
  }

  grid_points << lowr_ext, inner_grid, uppr_ext;
  return grid_points;
}

//! moves the boundary points of the grid to xmin/xmax (if non-NaN).
//! @param grid_points the grid points.
inline Eigen::VectorXd LPDens1d::finalize_grid(Eigen::VectorXd &grid_points) {
  if (!std::isnan(xmin_))
    grid_points(0) = xmin_;
  if (!std::isnan(xmax_))
    grid_points(grid_points.size() - 1) = xmax_;

  return grid_points;
}

//! removes the boundary extension from the grid_points (see
//! `construct_grid_points`).
//! @param grid_points the grid points.
inline Eigen::VectorXd
LPDens1d::without_boundary_ext(const Eigen::VectorXd &grid_points) {
  size_t grid_start = 0;
  size_t grid_size = grid_points.size();
  // (grid extension has length 2)
  if (std::isnan(xmin_)) {
    grid_start += 1;
    grid_size -= 2;
  }
  if (std::isnan(xmax_))
    grid_size -= 2;

  return grid_points.segment(grid_start, grid_size);
}
