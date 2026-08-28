test_that("installed C++ headers remain consumable", {
  if (dir.exists(test_path("..", "..", "inst", "include"))) {
    old_cppflags <- Sys.getenv("PKG_CPPFLAGS", unset = NA_character_)
    on.exit(if (is.na(old_cppflags)) {
      Sys.unsetenv("PKG_CPPFLAGS")
    } else {
      Sys.setenv(PKG_CPPFLAGS = old_cppflags)
    })
    Sys.setenv(PKG_CPPFLAGS = paste(
      "-I",
      shQuote(normalizePath(test_path("..", "..", "inst", "include")))
    ))
  }
  Rcpp::sourceCpp(test_path("cpp", "headers.cpp"), env = environment())
  expect_true(kde1d_headers_compile())
})
