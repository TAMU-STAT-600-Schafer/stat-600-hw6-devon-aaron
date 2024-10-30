
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)

# Source your C++ funcitons
# Check wd: getwd()
sourceCpp("./src/LRMultiClass.cpp")

# Source your LASSO functions from HW3 (make sure to move the corresponding .R file in the current project folder)
source("./R/LR_wrapper.R")


###


##########################
# Test helper functions: #
##########################

# One hot encoding:
# uvec_one_hot(as.vector(c(1, 3, 2, 1, 1, 2)), 6, 3) # Does not run passing in R vector objects (indexing issue)
# Diagonal sum:
sum_diag(matrix(c(1, 0, 0, 2), byrow = TRUE, ncol = 2), num_col = 2)


###

########################
# Test LRMultiClass_c: #
########################

# Application of multi-class logistic to letters data

# Load the letter data
#########################
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# [ToDo] Make sure to add column for an intercept to X
X <- cbind(1, X)

# [ToDo] Include beta_init object (default beta_init in LRMultiClass R function):
beta_init <- matrix(0, nrow = dim(X)[2], ncol = length(unique(Y))) # ncol: K

# [ToDo] Try the algorithm LRMultiClass with lambda = 1 and 50 iterations. Call the resulting object out, i.e. out <- LRMultiClass(...)
out <- LRMultiClass_c(X, Y, beta_init)
out2 <- LRMultiClass_c(X, Y, beta_init, numIter = 1)
out
out2

# # Alternative inputs:
# # High lambda (regularization)
# out <- LRMultiClass(X, Y, Xt, Yt, lambda = 5)
# # Low lambda
# out <- LRMultiClass(X, Y, Xt, Yt, lambda = 0.1)
# # High eta (step size)
# out <- LRMultiClass(X, Y, Xt, Yt, eta = 0.8)
# # Low eta
# out <- LRMultiClass(X, Y, Xt, Yt, eta = 0.001)


# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out$objective, type = 'o')

# Feel free to modify the code above for different lambda/eta/numIter values to see how it affects the convergence as well as train/test errors

# [ToDo] Use microbenchmark to time your code with lambda=1 and 50 iterations. To save time, only apply microbenchmark 5 times.
library(microbenchmark)
check_runtime <- function(num_times) {
  benchmark <- microbenchmark(
    
    LRMultiClass_cpp = LRMultiClass_c(X, Y, beta_init),
    times = num_times
    
  )
  return(benchmark)
}
check_runtime(num_times = 5L)

# [ToDo] Report the median time of your code from microbenchmark above in the comments below

# Median time: 8.142604 seconds (Intel chip)


# Debugger:
# Debug LRMultiClass_c:
debug(LRMultiClass_c)
LRMultiClass_c(X, Y, beta_init)
undebug(LRMultiClass_c)




