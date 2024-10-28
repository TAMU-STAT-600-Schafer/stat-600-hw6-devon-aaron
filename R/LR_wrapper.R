
#' Title
#'
#' @param X 
#' @param y 
#' @param numIter 
#' @param eta 
#' @param lambda 
#' @param beta_init 
#'
#' @return
#' @export
#'
#' @examples
#' # Give example
LRMultiClass <- function(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  
  # Compatibility checks from HW3 and initialization of beta_init
  
  ## Check the supplied parameters as described. You can assume that X is matrix; y is vector; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  
  # Initialize K, p, n:
  K <- length(unique(y))
  p <- dim(X)[2]
  n <- length(y)
  # Initialize vector objects (errors/objective):
  error_train <- vector()
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
  out = LRMultiClass_c(X, y, numIter, eta, lambda, beta_init)
  
  # Return the class assignments
  return(out)
}