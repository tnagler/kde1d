#!/bin/bash

git clone --depth 1 git@github.com:vinecopulib/kde1d-cpp.git -b main  --single-branch

rm -rf ../inst/include/kde1d/*
mv ./kde1d-cpp/include/* ../inst/include

rm -rf ./../inst/include/kde1d/mainpage.h
rm -rf kde1d-cpp
