#pragma once

#include "dpik.hpp"
#include "interpolation.hpp"
#include "stats.hpp"
#include "tools.hpp"
#include <cmath>
#include <functional>

namespace kde1d {

enum class VarType
{
  continuous,
  discrete,
  zero_inflated
};

//! Local-polynomial density estimation in 1-d.
class Kde1d
{
public:
  // constructors
  Kde1d(double xmin,
        double xmax,
        VarType type,
        double multiplier = 1.0,
        double bandwidth = NAN,
        size_t degree = 2);

  Kde1d(double xmin = NAN,
        double xmax = NAN,
        std::string type = "continuous",
        double multiplier = 1.0,
        double bandwidth = NAN,
        size_t degree = 2);


  Kde1d(const interp::InterpolationGrid& grid,
        double xmin,
        double xmax,
        VarType type,
        double prob0_ = 0.0);

  Kde1d(const interp::InterpolationGrid& grid,
        double xmin = NAN,
        double xmax = NAN,
        std::string type = "continuous",
        double prob0_ = 0.0);


  // old API, to be deprecated after this version
  Kde1d(const Eigen::VectorXd& x,
        size_t nlevels = 0,
        double bw = NAN,
        double mult = 1.0,
        double xmin = NAN,
        double xmax = NAN,
        size_t deg = 2,
        const Eigen::VectorXd& weights = Eigen::VectorXd());


  void fit(const Eigen::VectorXd& x,
           const Eigen::VectorXd& weights = Eigen::VectorXd());

  // statistical functions
  Eigen::VectorXd pdf(const Eigen::VectorXd& x,
                      const bool& check_fitted = true) const;
  Eigen::VectorXd cdf(const Eigen::VectorXd& x,
                      const bool& check_fitted = true) const;
  Eigen::VectorXd quantile(const Eigen::VectorXd& x,
                           const bool& check_fitted = true) const;
  Eigen::VectorXd simulate(size_t n,
                           const std::vector<int>& seeds = {},
                           const bool& check_fitted = true) const;

  // getters
  Eigen::VectorXd get_values() const { return grid_.get_values(); }
  Eigen::VectorXd get_grid_points() const { return grid_.get_grid_points(); }
  double get_xmin() const { return xmin_; }
  double get_xmax() const { return xmax_; }
  VarType get_type() const { return type_; }
  std::string get_type_str() const { return this->as_str(type_); }
  double get_prob0() const { return prob0_; }
  double get_multiplier() const { return multiplier_; }
  double get_bandwidth() const { return bandwidth_; }
  size_t get_degree() const { return degree_; }
  double get_edf() const { return edf_; }
  double get_loglik() const { return loglik_; }
  void set_xmin_xmax(double xmin = NAN, double xmax = NAN);

  std::string str() const
  {
    std::stringstream ss;
    ss << "Kde1d("
       << "xmin=" << xmin_ << ", xmax=" << xmax_ << ", type='"
       << this->as_str(type_) << "'"
       << ", bandwidth=" << bandwidth_ << ", multiplier=" << multiplier_
       << ", degree=" << degree_ << ")";
    return ss.str();
  }

protected:
  void set_interpolation_grid(const interp::InterpolationGrid& grid);

private:
  // data members
  interp::InterpolationGrid grid_;
  double xmin_;
  double xmax_;
  VarType type_;
  double multiplier_;
  double bandwidth_;
  size_t degree_;
  double prob0_{ 0.0 };
  double loglik_{ NAN };
  double edf_{ NAN };
  static constexpr double K0_ = 0.3989425;

  // private methods
  void check_fitted() const;
  void check_notfitted() const;
  void check_xmin_xmax(const double& xmin, const double& xmax) const;
  void check_inputs(const Eigen::VectorXd& x,
                    const Eigen::VectorXd& weights = Eigen::VectorXd()) const;
  void check_boundaries(const Eigen::VectorXd& x) const;
  Eigen::VectorXd pdf_continuous(const Eigen::VectorXd& x) const;
  Eigen::VectorXd cdf_continuous(const Eigen::VectorXd& x) const;
  Eigen::VectorXd quantile_continuous(const Eigen::VectorXd& x) const;
  Eigen::VectorXd pdf_discrete(const Eigen::VectorXd& x) const;
  Eigen::VectorXd cdf_discrete(const Eigen::VectorXd& x) const;
  Eigen::VectorXd quantile_discrete(const Eigen::VectorXd& x) const;
  Eigen::VectorXd pdf_zi(const Eigen::VectorXd& x) const;
  Eigen::VectorXd cdf_zi(const Eigen::VectorXd& x) const;
  Eigen::VectorXd quantile_zi(const Eigen::VectorXd& x) const;

  Eigen::VectorXd kern_gauss(const Eigen::VectorXd& x);
  Eigen::MatrixXd fit_lp(const Eigen::VectorXd& x,
                         const Eigen::VectorXd& grid,
                         const Eigen::VectorXd& weights);
  double calculate_infl(const size_t& n,
                        const double& f0,
                        const double& b,
                        const double& bandwidth,
                        const double& s,
                        const double& weight);
  Eigen::VectorXd boundary_transform(const Eigen::VectorXd& x,
                                     bool inverse = false);
  Eigen::VectorXd boundary_correct(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& fhat);
  Eigen::VectorXd construct_grid_points(const Eigen::VectorXd& x);
  Eigen::VectorXd finalize_grid(Eigen::VectorXd& grid_points);
  double select_bandwidth(const Eigen::VectorXd& x,
                          double bandwidth,
                          double multiplier,
                          size_t degree,
                          const Eigen::VectorXd& weights) const;

  std::string as_str(VarType type) const;
  VarType as_enum(std::string type) const;
};

//! constructor for fitting the density estimate.
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
//! @param type variable type: `VarType::continuous`  for
//!   continuous variables, `VarType::discrete` for discrete integer
//!   variables, or `VarType::zero_inflated` for zero-inflated
//!   variables.
//! @param multiplier bandwidth multiplier (default is 1.0).
//! @param bandwidth positive bandwidth parameter (`NaN` means automatic
//! selection).
//! @param degree degree of the local polynomial.
inline Kde1d::Kde1d(double xmin,
                    double xmax,
                    VarType type,
                    double multiplier,
                    double bandwidth,
                    size_t degree)
  : xmin_(xmin)
  , xmax_(xmax)
  , type_(type)
  , multiplier_(multiplier)
  , bandwidth_(bandwidth)
  , degree_(degree)
{
  this->check_xmin_xmax(xmin, xmax);
  if (multiplier <= 0.0) {
    throw std::invalid_argument("multiplier must be positive");
  }
  if (!std::isnan(bandwidth_) && (bandwidth_ <= 0.0)) {
    throw std::invalid_argument("bandwidth must be positive");
  }
  if (degree_ > 2) {
    throw std::invalid_argument("degree must be 0, 1 or 2");
  }
}

//! construct model from an already fit interpolation grid.
//! @param grid the interpolation grid.
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
//! @param type variable type: `VarType::continuous`  for
//!   continuous variables, `VarType::discrete` for discrete integer
//!   variables, or `VarType::zero_inflated` for zero-inflated
//!   variables.
//! @param prob0 point mass at 0.
inline Kde1d::Kde1d(const interp::InterpolationGrid& grid,
                    double xmin,
                    double xmax,
                    VarType type,
                    double prob0)
  : grid_(grid)
  , xmin_(xmin)
  , xmax_(xmax)
  , type_(type)
  , prob0_(prob0)
{
  this->check_xmin_xmax(xmin, xmax);
  if ((prob0 < 0) || (prob0 > 1)) {
    throw std::invalid_argument("prob0 must lie in the interval [0, 1].");
  }
}

//! constructor for fitting the density estimate.
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
//! @param type variable type; must be one of {"c", "cont", "continuous"} for
//!   continuous variables, one of {"d", "disc", "discrete"} for discrete
//!   integer variables, or one of {"zi", "zinfl", "zero-inflated"} for
//!   zero-inflated variables.
//! @param multiplier bandwidth multiplier (default is 1.0).
//! @param bandwidth positive bandwidth parameter (`NaN` means automatic
//! selection).
//! @param degree degree of the local polynomial.
inline Kde1d::Kde1d(double xmin,
                    double xmax,
                    std::string type,
                    double multiplier,
                    double bandwidth,
                    size_t degree)
  : Kde1d(xmin, xmax, this->as_enum(type), multiplier, bandwidth, degree)
{
}

//! construct model from an already fit interpolation grid.
//! @param grid the interpolation grid.
//! @param xmin lower bound for the support of the density, `NaN` means no
//!   boundary.
//! @param xmax upper bound for the support of the density, `NaN` means no
//!   boundary.
//! @param type variable type; must be one of {"c", "cont", "continuous"} for
//!   continuous variables, one of {"d", "disc", "discrete"} for discrete
//!   integer variables, or one of {"zi", "zinfl", "zero-inflated"} for
//!   zero-inflated variables.
//! @param prob0 point mass at 0.
inline Kde1d::Kde1d(const interp::InterpolationGrid& grid,
                    double xmin,
                    double xmax,
                    std::string type,
                    double prob0)
  : Kde1d(grid, xmin, xmax, this->as_enum(type), prob0)
{
}

// old API, to be deprecated after this version
Kde1d::Kde1d(const Eigen::VectorXd& x,
             size_t nlevels,
             double bw,
             double mult,
             double xmin,
             double xmax,
             size_t deg,
             const Eigen::VectorXd& weights)
  : Kde1d(xmin, xmax, nlevels > 0 ? VarType::discrete : VarType::continuous,
    mult, bw, deg)
{
  this->fit(x, weights);
}


//! @param x vector of observations
//! @param weights vector of weights for each observation (optional).
inline void
Kde1d::fit(const Eigen::VectorXd& x, const Eigen::VectorXd& weights)
{
  check_inputs(x, weights);
  check_boundaries(x);

  // preprocessing for nans and jittering
  Eigen::VectorXd xx = x;
  Eigen::VectorXd w = weights;
  tools::remove_nans(xx, w);

  if (w.size() > 0)
    w /= w.mean();

  if (type_ == VarType::zero_inflated) {
    if (w.size() == 0)
      w = Eigen::VectorXd::Ones(xx.size());
    w = (xx.array() == 0.0).select(Eigen::VectorXd::Zero(xx.size()), w);
    prob0_ = 1 - w.mean();
    xx =
      (w.array() == 0.0).select(Eigen::VectorXd::Constant(xx.size(), NAN), xx);
    tools::remove_nans(xx, w);
    if (xx.size() == 0) {
      bandwidth_ = NAN;
      loglik_ = 0.0;
      edf_ = 1.0;
      Eigen::VectorXd grid_points(5);
      grid_points << -2, -1, 0, 1, 2;
      auto values = Eigen::VectorXd::Constant(5, 0.0);
      grid_ = interp::InterpolationGrid(grid_points, values, 0);
      return;
    }
  } else if (type_ == VarType::discrete) {
    xx = stats::equi_jitter(xx);
  }

  xx = boundary_transform(xx);

  // bandwidth selection
  bandwidth_ = select_bandwidth(xx, bandwidth_, multiplier_, degree_, w);

  // fit model and evaluate in transformed domain
  Eigen::VectorXd grid_points = construct_grid_points(xx);
  Eigen::MatrixXd fitted = fit_lp(xx, boundary_transform(grid_points), w);

  // correct estimated density for transformation
  Eigen::VectorXd values = boundary_correct(grid_points, fitted.col(0));

  // move boundary points to xmin/xmax
  grid_points = finalize_grid(grid_points);

  // construct interpolation grid
  // (3 iterations for normalization to a proper density)
  grid_ = interp::InterpolationGrid(grid_points, values, 3);

  // calculate log-likelihood of final estimate
  xx = boundary_transform(xx, true);
  if (type_ == VarType::discrete) {
    xx = xx.array().round();
  }
  loglik_ = (this->pdf(xx, false).array().log()).sum();

  // calculate effective degrees of freedom
  interp::InterpolationGrid infl_grid(
      grid_points, fitted.col(1).cwiseMin(2.0).cwiseMax(0), 0);
  Eigen::VectorXd influences = infl_grid.interpolate(xx).array() * (1 - prob0_);
  edf_ = influences.sum() + (prob0_ > 0);

  // store bandwidth in standardized format
  bandwidth_ = bandwidth_ / multiplier_;
}

//! computes the pdf of the kernel density estimate by interpolation.
//! @param x vector of evaluation points.
//! @param check_fitted an optional logical to bypass the check.
//! @return a vector of pdf values.
inline Eigen::VectorXd
Kde1d::pdf(const Eigen::VectorXd& x, const bool& check_fitted) const
{
  if (check_fitted == true) {
    this->check_fitted();
  }
  check_inputs(x);

  switch (type_) {
  default:
    return pdf_continuous(x);
  case VarType::discrete:
    return pdf_discrete(x);
  case VarType::zero_inflated:
    return pdf_zi(x);
  }
}

inline Eigen::VectorXd
Kde1d::pdf_continuous(const Eigen::VectorXd& x) const
{
  Eigen::VectorXd fhat = grid_.interpolate(x);
  auto trunc = [](const double& xx) { return std::max(xx, 0.0); };
  return tools::unaryExpr_or_nan(fhat, trunc);
}

inline Eigen::VectorXd
Kde1d::pdf_discrete(const Eigen::VectorXd& x) const
{
  auto fhat = pdf_continuous(x);
  auto lb = std::floor(grid_.get_grid_min());
  auto ub = std::ceil(grid_.get_grid_max());
  Eigen::VectorXd lvs =
    Eigen::VectorXd::LinSpaced(static_cast<size_t>(ub - lb + 1), lb, ub);

  auto selected =
    (x.array() >= lb) && (x.array() <= ub) && (x.array() == x.array().round());
  fhat = fhat.array() * selected.cast<double>().array();

  // normalize
  fhat /= grid_.interpolate(lvs).sum();

  return fhat;
}

inline Eigen::VectorXd
Kde1d::pdf_zi(const Eigen::VectorXd& x) const
{
  auto ones = Eigen::VectorXd::Ones(x.size());
  return (x.array() == 0)
           .select(prob0_ * ones.array(), (1 - prob0_) * pdf_continuous(x).array());
}

//! computes the cdf of the kernel density estimate by numerical
//! integration.
//! @param x vector of evaluation points.
//! @param check_fitted an optional logical to bypass the check.
//! @return a vector of cdf values.
inline Eigen::VectorXd
Kde1d::cdf(const Eigen::VectorXd& x, const bool& check_fitted) const
{
  if (check_fitted == true) {
    this->check_fitted();
  }
  check_inputs(x);

  switch (type_) {
  default:
    return cdf_continuous(x);
  case VarType::discrete:
    return cdf_discrete(x);
  case VarType::zero_inflated:
    return cdf_zi(x);
  }
}

inline Eigen::VectorXd
Kde1d::cdf_continuous(const Eigen::VectorXd& x) const
{
  return grid_.integrate(x, /* normalize */ true);
}

inline Eigen::VectorXd
Kde1d::cdf_discrete(const Eigen::VectorXd& x) const
{
  auto lb = std::floor(grid_.get_grid_min());
  auto ub = std::ceil(grid_.get_grid_max());
  Eigen::VectorXd lvs =
    Eigen::VectorXd::LinSpaced(static_cast<size_t>(ub - lb + 1), lb, ub);

  auto f_cum = pdf_discrete(lvs);
  for (Eigen::Index i = 1; i < f_cum.size(); ++i)
    f_cum(i) += f_cum(i - 1);

  return tools::unaryExpr_or_nan(x, [&](const double& xx) {
    if (xx < lb) {
      return 0.0;
    } else if (xx >= ub) {
      return 1.0;
    } else {
      return f_cum(static_cast<size_t>(xx - lb));
    };
  });
}

inline Eigen::VectorXd
Kde1d::cdf_zi(const Eigen::VectorXd& x) const
{
  auto ones = Eigen::VectorXd::Ones(x.size());
  auto zeros = Eigen::VectorXd::Zero(x.size());
  Eigen::VectorXd zi = (x.array() >= 0).array().select(ones, zeros);
  return prob0_ * zi + (1 - prob0_) * (prob0_ < 1 ? cdf_continuous(x) : zeros);
}

//! computes the cdf of the kernel density estimate by numerical inversion.
//! @param x vector of evaluation points.
//! @param check_fitted an optional logical to bypass the check.
//! @return a vector of quantiles.
inline Eigen::VectorXd
Kde1d::quantile(const Eigen::VectorXd& x, const bool& check_fitted) const
{
  if (check_fitted == true) {
    this->check_fitted();
  }
  if ((x.minCoeff() < 0) || (x.maxCoeff() > 1))
    throw std::invalid_argument("probabilities must lie in (0, 1).");

  switch (type_) {
  default:
    return quantile_continuous(x);
  case VarType::discrete:
    return quantile_discrete(x);
  case VarType::zero_inflated:
    return quantile_zi(x);
  }
}

inline Eigen::VectorXd
Kde1d::quantile_continuous(const Eigen::VectorXd& x) const
{
  auto cdf = [&](const Eigen::VectorXd& xx) { return grid_.integrate(xx); };
  auto q =
    tools::invert_f(x, cdf, grid_.get_grid_min(), grid_.get_grid_max(), 35);

  // replace with NaN where the input was NaN
  for (long i = 0; i < x.size(); i++) {
    if (std::isnan(x(i)))
      q(i) = x(i);
  }

  return q;
}

inline Eigen::VectorXd
Kde1d::quantile_discrete(const Eigen::VectorXd& x) const
{
  auto lb = std::floor(grid_.get_grid_min());
  auto ub = std::ceil(grid_.get_grid_max());
  auto nlevels = static_cast<size_t>(ub - lb + 1);
  Eigen::VectorXd lvs = Eigen::VectorXd::LinSpaced(nlevels, lb, ub);

  auto p = cdf_discrete(lvs);
  auto quan = [&](const double& pp) {
    size_t lv = 0;
    while ((pp >= p(lv)) && (lv < nlevels - 1))
      lv++;
    return lvs(lv);
  };

  return tools::unaryExpr_or_nan(x, quan);
}

inline Eigen::VectorXd
Kde1d::quantile_zi(const Eigen::VectorXd& x) const
{
  Eigen::VectorXd qs(x.size());
  auto p0 = this->cdf(Eigen::VectorXd::Zero(1), false)(0);
  auto newx = (x.array() <= p0 - prob0_)
                .select(x / (1 - prob0_),
  (x.array() - prob0_).cwiseMax(0.0) / (1 - prob0_));
  qs = this->quantile_continuous(newx);
  for (Eigen::Index i = 0; i < x.size(); i++) {
    if ((x(i) > p0 - prob0_) && (x(i) <= p0)) {
      qs(i) = 0;
    }
  }
  return qs;
}

//! simulates data from the model.
//! @param n the number of observations to simulate.
//! @param seeds an optional vector of seeds.
//! @param check_fitted an optional logical to bypass the check.
//! @return simulated observations from the kernel density.
inline Eigen::VectorXd
Kde1d::simulate(size_t n,
                const std::vector<int>& seeds,
                const bool& check_fitted) const
{
  if (check_fitted == true) {
    this->check_fitted();
  }
  auto u = stats::simulate_uniform(n, seeds);
  return this->quantile(u);
}

//! Gaussian kernel (truncated at +/- 5).
//! @param x vector of evaluation points.
inline Eigen::VectorXd
Kde1d::kern_gauss(const Eigen::VectorXd& x)
{
  auto f = [](double xx) {
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
inline Eigen::MatrixXd
Kde1d::fit_lp(const Eigen::VectorXd& x,
              const Eigen::VectorXd& grid_points,
              const Eigen::VectorXd& weights)
{
  size_t m = grid_points.size();
  fft::KdeFFT kde_fft(
      x, bandwidth_, grid_points(0), grid_points(m - 1), weights);
  Eigen::VectorXd f0 = kde_fft.kde_drv(0);

  Eigen::VectorXd wbin = Eigen::VectorXd::Ones(m);
  if (weights.size()) {
    // compute the average weight per cell
    auto wcount = kde_fft.get_bin_counts();
    auto count = tools::linbin(x,
                               grid_points(0),
                               grid_points(m - 1),
                               m - 1,
                               Eigen::VectorXd::Ones(x.size()));
    wbin = wcount.cwiseQuotient(count);
  }

  Eigen::MatrixXd res(f0.size(), 2);
  res.col(0) = f0;
  res.col(1) =
    K0_ / (static_cast<double>(x.size()) * bandwidth_) * wbin.cwiseQuotient(f0);
  if (degree_ == 0)
    return res;

  // degree > 0
  Eigen::VectorXd f1 = kde_fft.kde_drv(1);
  Eigen::VectorXd S = Eigen::VectorXd::Constant(f0.size(), bandwidth_);
  Eigen::VectorXd b = f1.cwiseQuotient(f0);
  if (degree_ == 2) {
    Eigen::VectorXd f2 = kde_fft.kde_drv(2);
    // D/R is notation from Hjort and Jones' AoS paper
    Eigen::VectorXd D = f2.cwiseQuotient(f0) - b.cwiseProduct(b);
    Eigen::VectorXd R = 1 / (1.0 + bandwidth_ * bandwidth_ * D.array()).sqrt();
    // this is our notation
    S = (R / bandwidth_).array().pow(2);
    b *= bandwidth_ * bandwidth_;
    res.col(0) = bandwidth_ * S.cwiseSqrt().cwiseProduct(res.col(0));
  }
  res.col(0) = res.col(0).array() * (-0.5 * b.array().pow(2) * S.array()).exp();

  for (size_t k = 0; k < m; k++) {
    // TODO: weights
    res(k, 1) =
      calculate_infl(x.size(), f0(k), b(k), bandwidth_, S(k), wbin(k));
    if (std::isnan(res(k, 0)))
      res.row(k).setZero();
  }

  return res;
}

//! calculate influence for data point for density estimate based on
//! quantities pre-computed in `fit_lp()`.
inline double
  Kde1d::calculate_infl(const size_t& n,
                        const double& f0,
                        const double& b,
                        const double& bandwidth,
                        const double& s,
                        const double& weight)
  {
    double M_inverse00;
    double bandwidth2 = std::pow(bandwidth, 2);
    double b2 = std::pow(b, 2);
    if (degree_ == 0) {
      M_inverse00 = 1 / f0;
    } else if (degree_ == 1) {
      Eigen::Matrix2d M;
      M(0, 0) = f0;
      M(0, 1) = bandwidth2 * b * f0;
      M(1, 0) = M(0, 1);
      M(1, 1) = f0 * bandwidth2 + f0 * bandwidth2 * bandwidth2 * b2;
      M_inverse00 = M.inverse()(0, 0);
    } else {
      Eigen::Matrix3d M;
      M(0, 0) = f0;
      M(0, 1) = f0 * b;
      M(1, 0) = M(0, 1);
      M(1, 1) = f0 * bandwidth2 + f0 * b2;
      M(1, 2) = 0.5 * f0 * (3.0 / s * b + b * b2);
      M(2, 1) = M(1, 2);
      M(2, 2) = 0.25 * f0;
      M(2, 2) *= 3.0 / std::pow(s, 2) + 6.0 / s * b2 + b2 * b2;
      M(0, 2) = M(2, 2);
      M(2, 0) = M(2, 2);
      M_inverse00 = M.inverse()(0, 0);
    }

    return K0_ * weight / (static_cast<double>(n) * bandwidth) * M_inverse00;
  }

//! transformations for density estimates with bounded support.
//! @param x evaluation points.
//! @param inverse whether the inverse transformation should be applied.
//! @return the transformed evaluation points.
inline Eigen::VectorXd
Kde1d::boundary_transform(const Eigen::VectorXd& x, bool inverse)
{
  if (type_ == VarType::discrete) {
    return x; // no transform for discrete variables
  }

  Eigen::VectorXd x_new = x;
  if (!inverse) {
    if (!std::isnan(xmin_) && !std::isnan(xmax_)) {
      // two boundaries -> probit transform
      auto rng = xmax_ - xmin_;
      x_new = (x.array() - xmin_ + 5e-5 * rng) / (1.0001 * rng);
      x_new = stats::qnorm(x_new);
    } else if (!std::isnan(xmin_)) {
      // left boundary -> log transform
      x_new = (1e-5 + x.array() - xmin_).log();
    } else if (!std::isnan(xmax_)) {
      // right boundary -> negative log transform
      x_new = (1e-5 + xmax_ - x.array()).log();
    } else {
      // no boundary -> no transform
    }
  } else {
    if (!std::isnan(xmin_) && !std::isnan(xmax_)) {
      // two boundaries -> probit transform
      auto rng = xmax_ - xmin_;
      x_new = stats::pnorm(x).array() * 1.0001 * rng + xmin_ - 5e-5 * rng;
    } else if (!std::isnan(xmin_)) {
      // left boundary -> log transform
      x_new = x.array().exp() + xmin_ - 1e-5;
    } else if (!std::isnan(xmax_)) {
      // right boundary -> negative log transform
      x_new = -(x.array().exp() - xmax_ - 1e-5);
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
inline Eigen::VectorXd
Kde1d::boundary_correct(const Eigen::VectorXd& x, const Eigen::VectorXd& fhat)
{
  if (type_ == VarType::discrete) {
    return fhat; // no transform for discrete variables
  }

  Eigen::VectorXd corr_term(fhat.size());
  if (!std::isnan(xmin_) && !std::isnan(xmax_)) {
    // two boundaries -> probit transform
    auto rng = xmax_ - xmin_;
    corr_term = (x.array() - xmin_ + 5e-5 * rng) / (xmax_ - xmin_ + 1e-4 * rng);
    corr_term = stats::dnorm(stats::qnorm(corr_term));
    corr_term /= (xmax_ - xmin_ + 1e-4 * rng);
    corr_term = 1.0 / corr_term.array().max(1e-6);
  } else if (!std::isnan(xmin_)) {
    // left boundary -> log transform
    corr_term = 1.0 / (1e-5 + x.array() - xmin_).max(1e-6);
  } else if (!std::isnan(xmax_)) {
    // right boundary -> negative log transform
    corr_term = 1.0 / (1e-5 + xmax_ - x.array()).max(1e-6);
  } else {
    // no boundary -> no transform
    corr_term.fill(1.0);
  }

  Eigen::VectorXd f_corr = fhat.cwiseProduct(corr_term);
  if (std::isnan(xmin_) && !std::isnan(xmax_))
    f_corr.reverseInPlace();

  return f_corr;
}

//! constructs a grid later used for interpolation
//! @param x vector of observations.
//! @return a grid of size 50.
inline Eigen::VectorXd
Kde1d::construct_grid_points(const Eigen::VectorXd& x)
{
  Eigen::VectorXd rng(2);
  rng << x.minCoeff(), x.maxCoeff();
  if (std::isnan(xmin_) && std::isnan(xmax_)) {
    rng(0) -= 4 * bandwidth_;
    rng(1) += 4 * bandwidth_;
  }
  auto zgrid = Eigen::VectorXd::LinSpaced(401, rng(0), rng(1));
  return boundary_transform(zgrid, true);
}

//! moves the boundary points of the grid to xmin/xmax (if non-NaN).
//! @param grid_points the grid points.
inline Eigen::VectorXd
Kde1d::finalize_grid(Eigen::VectorXd& grid_points)
{
  if (std::isnan(xmin_) && !std::isnan(xmax_))
    grid_points.reverseInPlace();
  if (!std::isnan(xmin_))
    grid_points(0) = xmin_;
  if (!std::isnan(xmax_))
    grid_points(grid_points.size() - 1) = xmax_;

  return grid_points;
}

//  Bandwidth for Kernel Density Estimation
//' @param x vector of observations
 //' @param bandwidth bandwidth parameter, NA for automatic selection.
 //' @param multiplier bandwidth multiplieriplier.
 //' @param discrete whether a jittered estimate is computed.
 //' @param weights vector of weights for each observation (can be empty).
 //' @param degree polynomial degree.
 //' @return the selected bandwidth
 //' @noRd
 inline double
  Kde1d::select_bandwidth(const Eigen::VectorXd& x,
                          double bandwidth,
                          double multiplier,
                          size_t degree,
                          const Eigen::VectorXd& weights) const
  {
    if (std::isnan(bandwidth)) {
      bandwidth::PluginBandwidthSelector selector(x, weights);
      bandwidth = selector.select_bandwidth(degree);
    }

    bandwidth *= multiplier;
    if (type_ == VarType::discrete) {
      bandwidth = std::max(bandwidth, 0.5 / 5);
    }

    return bandwidth;
  }

inline void
Kde1d::check_xmin_xmax(const double& xmin, const double& xmax) const
{
  if (!std::isnan(xmax) && !std::isnan(xmax) && (xmin > xmax))
    throw std::invalid_argument("xmin must be smaller than xmax");
}

inline void
Kde1d::check_fitted() const
{
  if (std::isnan(loglik_)) {
    throw std::runtime_error("You must first fit the KDE to data.");
  }
}

inline void
Kde1d::check_notfitted() const
{
  if (!std::isnan(loglik_)) {
    throw std::runtime_error(
        "This method can't be used for already fitted objects.");
  }
}

inline void
Kde1d::check_inputs(const Eigen::VectorXd& x,
                    const Eigen::VectorXd& weights) const
{
  if (x.size() == 0)
    throw std::invalid_argument("x must not be empty");

  if ((weights.size() > 0) && (weights.size() != x.size()))
    throw std::invalid_argument("x and weights must have the same size");
}

inline void
Kde1d::check_boundaries(const Eigen::VectorXd& x) const
{
  if ((x.array() < xmin_).any() || (x.array() > xmax_).any()) {
    throw std::invalid_argument("x must be contained in [xmin, xmax].");
  }
}

void
Kde1d::set_interpolation_grid(const interp::InterpolationGrid& grid)
{
  grid_ = grid;
}

void
Kde1d::set_xmin_xmax(double xmin, double xmax)
{
  this->check_notfitted();
  this->check_xmin_xmax(xmin, xmax);
  xmin_ = xmin;
  xmax_ = xmax;
}

std::string
Kde1d::as_str(VarType type) const
{
  std::string type_str;
  switch (type) {
  case VarType::continuous:
    return "continuous";
  case VarType::discrete:
    return "discrete";
  case VarType::zero_inflated:
    return "zero-inflated";
  default:
    throw std::invalid_argument("unknown variable type.");
  }
}

VarType
Kde1d::as_enum(std::string type) const
{
  if ((type == "c") || (type == "cont") || (type == "continuous")) {
    return VarType::continuous;
  } else if ((type == "d") || (type == "disc") || (type == "discrete")) {
    return VarType::discrete;
  } else if ((type == "zi") || (type == "zinfl") || (type == "zero-inflated") ||
    (type == "zero_inflated")) {
    return VarType::zero_inflated;
  } else {
    std::stringstream ss;
    ss << "variable type '" << type << "' unknown; must be one of"
       << "{c, cont, continuous, d, disc, discrete, zi, zinfl, zero-inflated}."
       << std::endl;
    throw std::invalid_argument(ss.str());
  }
  return VarType::continuous;
}

} // end kde1d
