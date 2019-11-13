#pragma once

#include "kde1d/interpolation.hpp"
#include "kde1d/stats.hpp"
#include "kde1d/tools.hpp"
#include "kde1d/dpik.hpp"
#include <functional>
#include <cmath>

namespace kde1d {

//! Local-polynomial density estimation in 1-d.
class Kde1d {
public:
  // constructors
  Kde1d() {}
  Kde1d(const Eigen::VectorXd& x,
        size_t nlevels = 0,
        double bw = NAN,
        double mult = 1.0,
        double xmin = NAN,
        double xmax = NAN,
        size_t deg = 2,
        const Eigen::VectorXd& weights = Eigen::VectorXd());
  Kde1d(const interp::InterpolationGrid1d& grid,
        size_t nlevels = 0,
        double xmin = NAN,
        double xmax = NAN);

  // statistical functions
  Eigen::VectorXd pdf(const Eigen::VectorXd& x) const;
  Eigen::VectorXd cdf(const Eigen::VectorXd& x) const;
  Eigen::VectorXd quantile(const Eigen::VectorXd& x) const;
  Eigen::VectorXd simulate(size_t n,
                           const std::vector<int>& seeds = {}) const;

  // getters
  Eigen::VectorXd get_values() const { return grid_.get_values(); }
  Eigen::VectorXd get_grid_points() const { return grid_.get_grid_points(); }
  size_t get_nlevels() const { return nlevels_; }
  double get_bw() const {return bw_;}
  double get_deg() const {return deg_;}
  double get_xmin() const {return xmin_;}
  double get_xmax() const {return xmax_;}
  double get_edf() const {return edf_;}
  double get_loglik() const {return loglik_;}

private:
  // data members
  interp::InterpolationGrid1d grid_;
  size_t nlevels_;
  double xmin_;
  double xmax_;
  double bw_{NAN};
  size_t deg_{2};
  double loglik_{NAN};
  double edf_{NAN};
  static constexpr double K0_ = 0.3989425;

  // private methods
  Eigen::VectorXd pdf_continuous(const Eigen::VectorXd& x) const;
  Eigen::VectorXd cdf_continuous(const Eigen::VectorXd& x) const;
  Eigen::VectorXd quantile_continuous(const Eigen::VectorXd& x) const;
  Eigen::VectorXd pdf_discrete(const Eigen::VectorXd& x) const;
  Eigen::VectorXd cdf_discrete(const Eigen::VectorXd& x) const;
  Eigen::VectorXd quantile_discrete(const Eigen::VectorXd& x) const;

  void check_levels(const Eigen::VectorXd& x) const;
  Eigen::VectorXd kern_gauss(const Eigen::VectorXd& x);
  Eigen::MatrixXd fit_lp(const Eigen::VectorXd& x_ev,
                         const Eigen::VectorXd& x,
                         const Eigen::VectorXd& weights);
  double calculate_infl(const size_t& n,
                        const double& f0,
                        const double& b,
                        const double& bw,
                        const double& s,
                        const double& weight);
  Eigen::VectorXd boundary_transform(const Eigen::VectorXd& x,
                                     bool inverse = false);
  Eigen::VectorXd boundary_correct(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& fhat);
  Eigen::VectorXd construct_grid_points(const Eigen::VectorXd& x,
                                        const Eigen::VectorXd& weights);
  Eigen::VectorXd finalize_grid(Eigen::VectorXd& grid_points);
  Eigen::VectorXd without_boundary_ext(const Eigen::VectorXd& grid_points);
  double select_bw(const Eigen::VectorXd& x,
                   double bw, double mult, size_t deg, size_t nlevels,
                   const Eigen::VectorXd& weights) const;
};

//! constructor for fitting the density estimate.
//! @param nlevels number of factor levels; 0 for continuous variables.
//! @param x vector of observations
//! @param bw positive bandwidth parameter (fixed component).
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
//! @param deg order of the local polynomial.
//! @param weights vector of weights for each observation (can be empty).
inline Kde1d::Kde1d(const Eigen::VectorXd& x,
                    size_t nlevels,
                    double bw,
                    double mult,
                    double xmin,
                    double xmax,
                    size_t deg,
                    const Eigen::VectorXd& weights)
  : nlevels_(nlevels)
  , xmin_(xmin)
  , xmax_(xmax)
  , bw_(bw)
  , deg_(deg)
{
  if (weights.size() > 0 && (weights.size() != x.size()))
    throw std::runtime_error("x and weights must have the same size.");
  if (deg > 2)
    throw std::runtime_error("deg must not be larger than 2.");
  check_levels(x);
  if (nlevels_ > 0) {
    xmin = NAN;
    xmax = NAN;
  }

  // preprocessing for nans and jittering
  Eigen::VectorXd xx = x;
  Eigen::VectorXd w = weights;
  tools::remove_nans(xx, w);
  if (w.size() > 0)
    w /= w.sum();
  if (nlevels_ > 0)
    xx = stats::equi_jitter(xx);

  // bandwidth selection
  xx = boundary_transform(xx);
  bw_ = select_bw(xx, bw_, mult, deg, nlevels_, w);

  // construct grid on original domain
  Eigen::VectorXd grid_points = construct_grid_points(xx, w);
  grid_points = boundary_transform(grid_points);

  // fit model and evaluate in transformed domain
  Eigen::MatrixXd fitted = fit_lp(grid_points, xx, w);

  // back-transform grid to original domain
  grid_points = boundary_transform(grid_points, true);
  xx = boundary_transform(xx, true);

  // correct estimated density for transformation
  Eigen::VectorXd values = boundary_correct(grid_points, fitted.col(0));

  // move boundary points to xmin/xmax
  grid_points = finalize_grid(grid_points);

  // construct interpolation grid
  // (3 iterations for normalization to a proper density)
  grid_ = interp::InterpolationGrid1d(grid_points, values, 3);

  // store normalized values
  values = grid_.get_values();

  // calculate log-likelihood of final estimate
  loglik_ = grid_.interpolate(x).cwiseMax(1e-20).array().log().sum();

  // calculate effective degrees of freedom
  double n = x.size();
  interp::InterpolationGrid1d infl_grid(
      without_boundary_ext(grid_points),
      without_boundary_ext(fitted.col(1).cwiseMin(2.0).cwiseMax(0)), 0);
  edf_ = infl_grid.interpolate(x).sum();
}


//! construct model from an already fit interpolation grid.
//! @param grid the interpolation grid.
//! @param nlevels number of factor levels; 0 for continuous variables.
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
inline Kde1d::Kde1d(const interp::InterpolationGrid1d& grid,
                    size_t nlevels,
                    double xmin,
                    double xmax)
  : grid_(grid)
  , nlevels_(nlevels)
  , xmin_(xmin)
  , xmax_(xmax)
{}

//! computes the pdf of the kernel density estimate by interpolation.
//! @param x vector of evaluation points.
//! @return a vector of pdf values.
inline Eigen::VectorXd Kde1d::pdf(const Eigen::VectorXd& x) const
{
  return (nlevels_ == 0) ? pdf_continuous(x) : pdf_discrete(x);
}

inline Eigen::VectorXd Kde1d::pdf_continuous(const Eigen::VectorXd& x) const
{
  Eigen::VectorXd fhat = grid_.interpolate(x);
  if (!std::isnan(xmin_)) {
    fhat = (x.array() < xmin_).select(Eigen::VectorXd::Zero(x.size()), fhat);
  }
  if (!std::isnan(xmax_)) {
    fhat = (x.array() > xmax_).select(Eigen::VectorXd::Zero(x.size()), fhat);
  }

  auto trunc = [] (const double& xx) { return std::max(xx, 0.0); };
  return tools::unaryExpr_or_nan(fhat, trunc);;
}

inline Eigen::VectorXd Kde1d::pdf_discrete(const Eigen::VectorXd& x) const
{
  check_levels(x);
  auto fhat = pdf_continuous(x);
  // normalize
  Eigen::VectorXd lvs = Eigen::VectorXd::LinSpaced(nlevels_, 0, nlevels_ - 1);
  fhat /= grid_.interpolate(lvs).sum();
  return fhat;
}

//! computes the cdf of the kernel density estimate by numerical integration.
//! @param x vector of evaluation points.
//! @return a vector of cdf values.
inline Eigen::VectorXd Kde1d::cdf(const Eigen::VectorXd& x) const
{
  return (nlevels_ == 0) ? cdf_continuous(x) : cdf_discrete(x);
}

inline Eigen::VectorXd Kde1d::cdf_continuous(const Eigen::VectorXd& x) const
{
  auto p = grid_.integrate(x);
  auto trunc = [] (const double& xx) {
    return std::min(std::max(xx, 0.0), 1.0);
  };
  return tools::unaryExpr_or_nan(p, trunc);
}

inline Eigen::VectorXd Kde1d::cdf_discrete(const Eigen::VectorXd& x) const
{
  check_levels(x);
  Eigen::VectorXd lvs = Eigen::VectorXd::LinSpaced(nlevels_, 0, nlevels_ - 1);
  auto f_cum = pdf(lvs);
  for (size_t i = 1; i < nlevels_; ++i)
    f_cum(i) += f_cum(i - 1);

  return tools::unaryExpr_or_nan(x, [&f_cum] (const double& xx) {
    return f_cum(static_cast<size_t>(xx));
  });
}

//! computes the cdf of the kernel density estimate by numerical inversion.
//! @param x vector of evaluation points.
//! @return a vector of quantiles.
inline Eigen::VectorXd Kde1d::quantile(const Eigen::VectorXd& x) const
{
  if ((x.minCoeff() < 0) | (x.maxCoeff() > 1))
    throw std::runtime_error("probabilities must lie in (0, 1).");
  return (nlevels_ == 0) ? quantile_continuous(x) : quantile_discrete(x);
}

inline Eigen::VectorXd Kde1d::quantile_continuous(const Eigen::VectorXd& x) const
{
  auto cdf = [&] (const Eigen::VectorXd& xx) { return grid_.integrate(xx); };
  auto q = tools::invert_f(x,
                           cdf,
                           grid_.get_grid_points().minCoeff(),
                           grid_.get_grid_points().maxCoeff(),
                           35);

  // replace with NaN where the input was NaN
  for (size_t i = 0; i < x.size(); i++) {
    if (std::isnan(x(i)))
      q(i) = std::numeric_limits<double>::quiet_NaN();
  }

  return q;
}

inline Eigen::VectorXd Kde1d::quantile_discrete(const Eigen::VectorXd& x) const
{
  Eigen::VectorXd lvs = Eigen::VectorXd::LinSpaced(nlevels_, 0, nlevels_ - 1);
  auto p = cdf(lvs);
  auto quan = [&] (const double& pp) {
    size_t lv = 0;
    while ((pp >= p(lv)) && (lv < nlevels_ - 1))
      lv++;
    return lvs(lv);
  };

  return tools::unaryExpr_or_nan(x, quan);
}

//! simulates data from the model.
//! @param n the number of observations to simulate.
//! @param seeds an optional vector of seeds.
//! @return simulated observations from the kernel density.
inline Eigen::VectorXd Kde1d::simulate(size_t n,
                                       const std::vector<int>& seeds) const
{
  auto u = stats::simulate_uniform(n, seeds);
  return this->quantile(u);
}

inline void Kde1d::check_levels(const Eigen::VectorXd& x) const
{
  if (nlevels_ == 0)
    return;
  if ((x.array() != x.array().round()).any() | (x.minCoeff() < 0)) {
    throw std::runtime_error("x must only contain positive "
                             " integers when nlevels > 0.");
  }
  if (x.maxCoeff() > nlevels_) {
    throw std::runtime_error("maximum value of 'x' is larger than the "
                             "number of factor levels.");
  }
}

//! Gaussian kernel (truncated at +/- 5).
//! @param x vector of evaluation points.
inline Eigen::VectorXd Kde1d::kern_gauss(const Eigen::VectorXd& x)
{
  auto f = [] (double xx) {
    // truncate at +/- 5
    if (std::fabs(xx) > 5.0)
      return 0.0;
    // otherwise calculate normal pdf (orrect for truncation)
    return stats::dnorm(Eigen::VectorXd::Constant(1, xx))(0) / 0.999999426;
  };
  return x.unaryExpr(f);
}

//! (analytically) evaluates the kernel density estimate and its influence
//! function on a user-supplied grid.
//! @param x_ev evaluation points.
//! @param x observations.
//! @param weights vector of weights for each observation (can be empty).
//! @return a two-column matrix containing the density estimate in the first
//!   and the influence function in the second column.
inline Eigen::MatrixXd Kde1d::fit_lp(const Eigen::VectorXd& x_ev,
                                     const Eigen::VectorXd& x,
                                     const Eigen::VectorXd& weights)
{
  Eigen::MatrixXd res(x_ev.size(), 2);
  size_t n = x.size();

  double f0, f1, b;
  double s = bw_;
  double w0 = 1.0;
  Eigen::VectorXd xx(x.size());
  Eigen::VectorXd xx2(x.size());
  Eigen::VectorXd kernels(x.size());
  for (size_t k = 0; k < x_ev.size(); k++) {
    double s = bw_;
    // classical (local constant) kernel density estimate
    xx = (x.array() - x_ev(k)) / bw_;
    kernels = kern_gauss(xx) / bw_;
    if (weights.size() > 0)
      kernels = kernels.cwiseProduct(weights);
    f0 = kernels.mean();
    res(k, 0) = f0;

    // Before continuing with higher-order polynomials, check
    // (local constant) influence. If it is close to one, there is only one
    // observation contributing to the estimate and it is evaluated close to
    // it. To avoid numerical issues in this case, we just use the
    // local constant estimate.
    if (weights.size()) {
      // find weight corresponding to observation closest to x_ev(k)
      w0 = weights(tools::find_min_index(xx.array().abs()));
    }
    res(k, 1) = K0_ * w0 / (n * bw_) / f0;
    if (res(k, 1) > 0.95) {
      continue;
    }

    if (deg_ > 0) {
      // calculate b for local linear
      xx /= bw_;
      f1 = xx.cwiseProduct(kernels).mean(); // first order derivative
      b = f1 / f0;

      if (deg_ == 2) {
        // more calculations for local quadratic
        xx2 = xx.cwiseProduct(kernels) / (f0 * static_cast<double>(n));
        b *= std::pow(bw_, 2);
        s = 1.0 / (std::pow(bw_, 4) * xx.transpose() * xx2 - b*b);
        res(k, 0) *= bw_ * std::sqrt(s);
      }

      // final estimate
      res(k, 0) *= std::exp(-0.5 * std::pow(b, 2) * s);
      res(k, 1) = calculate_infl(n, f0, b, bw_, s, w0);

      if (std::isnan(res(k)) | std::isinf(res(k))) {
        // inverse operation might go wrong due to rounding when
        // true value is equal or close to zero
        res(k, 0) = 0.0;
        res(k, 1) = 0.0;
      }
    }

  }

  return res;
}

//! calculate influence for data point for density estimate based on
//! quantities pre-computed in `fit_lp()`.
inline double Kde1d::calculate_infl(const size_t &n,
                                    const double& f0,
                                    const double& b,
                                    const double& bw,
                                    const double& s,
                                    const double& weight)
{
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
    M(2, 2) *= 3.0 / std::pow(s, 2) + 6.0 / s * b2  + b2 * b2;
    M(0, 2) = M(2, 2);
    M(2, 0) = M(2, 2);
  }

  return K0_ * weight / (n * bw) * M.inverse()(0, 0);
}


//! transformations for density estimates with bounded support.
//! @param x evaluation points.
//! @param inverse whether the inverse transformation should be applied.
//! @return the transformed evaluation points.
inline Eigen::VectorXd Kde1d::boundary_transform(const Eigen::VectorXd& x,
                                                 bool inverse)
{
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
      x_new *=  (xmax_ - xmin_ + 1e-4);
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
inline Eigen::VectorXd Kde1d::boundary_correct(const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& fhat)
{
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
inline Eigen::VectorXd Kde1d::construct_grid_points(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& weights)
{
  // hybrid grid: quantiles and two equally spaced points between them
  auto qgrid = stats::quantile(
    x, Eigen::VectorXd::LinSpaced(14, 0, 1), weights);
  size_t grid_size = 53;
  Eigen::VectorXd inner_grid(grid_size);
  for (unsigned i = 0; i < qgrid.size() - 1; i++) {
    inner_grid.segment(i * 4, 5) =
      Eigen::VectorXd::LinSpaced(5, qgrid(i), qgrid(i + 1));
  }

  // extend grid where there's no boundary
  double range = inner_grid.maxCoeff() - inner_grid.minCoeff();
  Eigen::VectorXd lowr_ext, uppr_ext;
  if (std::isnan(xmin_)) {
    // no left boundary -> add a few points to the left
    lowr_ext = Eigen::VectorXd(2);
    lowr_ext[1] = inner_grid[0] - 1 * bw_;
    lowr_ext[0] = inner_grid[0] - 2 * bw_;
  }
  if (std::isnan(xmax_)) {
    // no right boundary -> add a few points to the right
    uppr_ext = Eigen::VectorXd(2);
    uppr_ext[0] = inner_grid[grid_size - 1] + 1 * bw_;
    uppr_ext[1] = inner_grid[grid_size - 1] + 2 * bw_;
  }

  Eigen::VectorXd grid_points(grid_size + uppr_ext.size() + lowr_ext.size());
  grid_points << lowr_ext, inner_grid, uppr_ext;
  return grid_points;
}

//! moves the boundary points of the grid to xmin/xmax (if non-NaN).
//! @param grid_points the grid points.
inline Eigen::VectorXd Kde1d::finalize_grid(Eigen::VectorXd& grid_points)
{
  if (!std::isnan(xmin_))
    grid_points(0) = xmin_;
  if (!std::isnan(xmax_))
    grid_points(grid_points.size() - 1) = xmax_;

  return grid_points;
}

//! removes the boundary extension from the grid_points (see
//! `construct_grid_points`).
//! @param grid_points the grid points.
inline Eigen::VectorXd Kde1d::without_boundary_ext(
  const Eigen::VectorXd& grid_points)
{
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

//  Bandwidth for Kernel Density Estimation
//' @param x vector of observations
//' @param bw bandwidth parameter, NA for automatic selection.
//' @param mult bandwidth multiplier.
//' @param discrete whether a jittered estimate is computed.
//' @param weights vector of weights for each observation (can be empty).
//' @param deg polynomial degree.
//' @return the selected bandwidth
//' @noRd
inline double Kde1d::select_bw(const Eigen::VectorXd& x,
                               double bw, double mult, size_t deg,
                               size_t nlevels,
                               const Eigen::VectorXd& weights) const
{
  if (std::isnan(bw)) {
    bw::PluginBandwidthSelector selector(x, weights);
    bw = selector.select_bw(deg);
  }

  bw *= mult;
  if (nlevels > 0) {
    bw = std::max(bw, 0.5 / 5);
  }

  return bw;
}

} // end kde1d
