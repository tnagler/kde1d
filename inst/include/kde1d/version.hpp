// Copyright © 2016-2023 Thomas Nagler and Thibault Vatter
//
// This file is part of the vinecopulib library and licensed under the terms of
// the MIT license. For a copy, see the LICENSE file in the root directory of
// vinecopulib or https://vinecopulib.github.io/vinecopulib/.

#pragma once

//
//  Caution, this is the only vinecopulib header that is guaranteed
//  to change with every vinecopulib release, including this header
//  will cause a recompile every time a new vinecopulib version is
//  released.
//
//  KDE1D_VERSION % 100 is the patch level
//  KDE1D_VERSION / 100 % 1000 is the minor version
//  KDE1D_VERSION / 100000 is the major version

#define KDE1D_VERSION 000101

//
//  KDE1D_LIB_VERSION must be defined to be the same as
//  KDE1D_VERSION but as a *string* in the form "x_y[_z]" where x is the
//  major version number, y is the minor version number, and z is the patch
//  level if not 0.

#define KDE1D_LIB_VERSION "0_1_1"
