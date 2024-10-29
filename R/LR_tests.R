
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)

# Source your C++ funcitons
# Check wd: getwd()
sourceCpp("./src/LRMultiClass.cpp")

# Source your LASSO functions from HW3 (make sure to move the corresponding .R file in the current project folder)
source("./R/LR_wrapper.R")

# Test helper functions:
uvec_one_hot(as.vector(c(1, 3, 2, 1, 1, 2)), 6, 3)
sum_diag(matrix(c(1, 0, 0, 2), byrow = TRUE, ncol = 2))



