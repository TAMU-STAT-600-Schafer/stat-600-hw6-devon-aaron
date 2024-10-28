
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)

# Source your C++ funcitons
# Check wd: getwd()
sourceCpp("./src/LRMultiClass.cpp")

# Source your LASSO functions from HW3 (make sure to move the corresponding .R file in the current project folder)
source("./R/LR_wrapper.R")
