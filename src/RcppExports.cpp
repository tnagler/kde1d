// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// fit_kde1d_cpp
Rcpp::List fit_kde1d_cpp(const Eigen::VectorXd& x, double bw, double xmin, double xmax);
RcppExport SEXP _kde1d_fit_kde1d_cpp(SEXP xSEXP, SEXP bwSEXP, SEXP xminSEXP, SEXP xmaxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
    Rcpp::traits::input_parameter< double >::type xmin(xminSEXP);
    Rcpp::traits::input_parameter< double >::type xmax(xmaxSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_kde1d_cpp(x, bw, xmin, xmax));
    return rcpp_result_gen;
END_RCPP
}
// dkde1d_cpp
Eigen::VectorXd dkde1d_cpp(const Eigen::VectorXd& x, const Rcpp::List& R_object);
RcppExport SEXP _kde1d_dkde1d_cpp(SEXP xSEXP, SEXP R_objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type R_object(R_objectSEXP);
    rcpp_result_gen = Rcpp::wrap(dkde1d_cpp(x, R_object));
    return rcpp_result_gen;
END_RCPP
}
// pkde1d_cpp
Eigen::VectorXd pkde1d_cpp(const Eigen::VectorXd& x, const Rcpp::List& R_object);
RcppExport SEXP _kde1d_pkde1d_cpp(SEXP xSEXP, SEXP R_objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type R_object(R_objectSEXP);
    rcpp_result_gen = Rcpp::wrap(pkde1d_cpp(x, R_object));
    return rcpp_result_gen;
END_RCPP
}
// qkde1d_cpp
Eigen::VectorXd qkde1d_cpp(const Eigen::VectorXd& x, const Rcpp::List& R_object);
RcppExport SEXP _kde1d_qkde1d_cpp(SEXP xSEXP, SEXP R_objectSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type R_object(R_objectSEXP);
    rcpp_result_gen = Rcpp::wrap(qkde1d_cpp(x, R_object));
    return rcpp_result_gen;
END_RCPP
}
// select_bw_cpp
double select_bw_cpp(const Eigen::VectorXd& x);
RcppExport SEXP _kde1d_select_bw_cpp(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(select_bw_cpp(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_kde1d_fit_kde1d_cpp", (DL_FUNC) &_kde1d_fit_kde1d_cpp, 4},
    {"_kde1d_dkde1d_cpp", (DL_FUNC) &_kde1d_dkde1d_cpp, 2},
    {"_kde1d_pkde1d_cpp", (DL_FUNC) &_kde1d_pkde1d_cpp, 2},
    {"_kde1d_qkde1d_cpp", (DL_FUNC) &_kde1d_qkde1d_cpp, 2},
    {"_kde1d_select_bw_cpp", (DL_FUNC) &_kde1d_select_bw_cpp, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_kde1d(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
