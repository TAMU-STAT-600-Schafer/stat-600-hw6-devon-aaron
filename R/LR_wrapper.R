#' Logist Regression Multi-Class Regression
#'
#' @param X A n x p matrix of numeric data
#' @param y A response vector of length n containing classifications
#' @param numIter (Optional) Number of iterations to perform Damped Newton's Method
#' @param eta (Optional) Control parameter for step size of Damped Newton's Method
#' @param lambda (Optional) Penalty parameter for l2-norm of Beta
#' @param beta_init (Optional) Initial Beta value to use as starting point for iterative solution
#'
#' @return A list with the elements
#' \item{beta}{ A n x K matrix that's the iterative solution to Multi-Class Logistic Regression with penalty lambda }
#' \item{fmin}{ A vector of length numIter + 1 containing initial objective function value and subsequent objective function values in each iteration }
#' @export
#'
#' @examples
#' 
#' ##### Example 1
#' set.seed(112)
#' 
#' n1 <- 25
#' n2 <- 25
#' 
#' X <- rbind(matrix(rnorm(n1, mean = -1, sd = .5), byrow = TRUE, nrow = n1), 
#'            matrix(rnorm(n2, mean = 1, sd = .5), byrow = TRUE, nrow = n2))
#'            
#' X <- cbind(1, X)
#' 
#' y <- c(rep(0, 25), rep(1, 25))
#' 
#' LRMultiClass_Solution <- LRMultiClass(X, y)
#' 
#' #####################
#' ##### Example 2
#' set.seed(112)
#' 
#' n1 <- 10
#' n2 <- 10
#' n3 <- 10
#' 
#' X <- rbind(matrix(rnorm(4*n1, mean = -2, sd = .75), byrow = TRUE, nrow = n1),
#'            matrix(rnorm(4*n2, sd = .75), byrow = TRUE, nrow = n2),
#'            matrix(rnorm(4*n3, mean = 2, sd = .75), byrow = TRUE, nrow = n3))
#' 
#' X <- cbind(1, X)
#' 
#' y <- c(rep(0, 10), rep(1, 10), rep(2, 10))
#' 
#' LRMultiClass_Solution <- LRMultiClass(X, y)
#' 
LRMultiClass <- function(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  
  # Compatibility checks from HW3 and initialization of beta_init
  
  ## Check the supplied parameters as described. You can assume that X is matrix; y is vector; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  
  # Initialize K, p, n:
  K <- length(unique(y))
  p <- dim(X)[2]
  n <- length(y)
  # Initialize vector objects (objective):
  objective <- vector()
  
  # Check that the first column of X are 1s, if not - display appropriate message and stop execution.
  if (!all(X[,1] == 1)){
    
    stop(paste("X contains values other than 1 in first column. Check and readjust."))
    
  }
  
  ###
  
  # Check for compatibility of dimensions between X and Y
  if (dim(X)[1] != n) {
    
    stop(paste("Dimensions between X and Y are not compatible. Check and readjust."))
    
  }
  
  ###
  
  # Check eta is positive
  if (eta <= 0){
    
    stop(paste("eta parameter is not positive. Readjust."))
    
  }
  
  ###
  
  # Check lambda is non-negative
  if (lambda < 0){
    
    stop(paste("lambda parameter is negative. Readjust."))
    
  }
  
  ###
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (all(is.null(beta_init)) | all(is.na(beta_init))) {
    
    beta_init <- matrix(0, nrow = dim(X)[2], ncol = K)
    
  }
  # If not all NULL/NA, check compatibility with X and K
  else {
    
    # If dimensions incompatible: stop job, print error statement
    if (dim(beta_init)[1] != p | dim(beta_init)[2] != K) {
      
      stop(paste("Dimensions of beta_init not compatible with p and/or K. Check and readjust."))
      
    }
    # If compatible, continue
    
  }
  
  #############################################################
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  
  # Return the class assignments
  return(out)
}