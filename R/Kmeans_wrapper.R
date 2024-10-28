#' Title
#'
#' @param X 
#' @param K 
#' @param M 
#' @param numIter 
#'
#' @return Explain return
#' @export
#'
#' @examples
#' # Give example
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  ############# Adversarial user checks ##################
  {
    # Coerce M to matrix if it's a data.frame
    if(is.data.frame(X)){
      X = as.matrix(X)
    }
    
    stopifnot(exprs = {
      # Check K and numIter are double single value
      is.double(K)
      is.double(numIter)
      length(as.integer(K)) == 1
      length(as.integer(numIter)) == 1
      !is.na(K)
      !is.na(numIter)
      
      # Check to make sure X makes sense
      is.matrix(X)
      !any(is.na(X))
      is.double(X)
    })
    
    # Check K input makes sense, also storing it as an integer
    if (length(as.integer(K)) != 1) {
      stop("K is not an integer")
    }
    
    if(floor(K) != K){
      warning("K is not an integer, floor(K) will be used")
    }
    
    K = as.integer(K)
    
    if (K <= 0) {
      stop("K is not a positive integer")
    }
    
    # Check numIter input makes sense, also storing it as an integer
    if (length(as.integer(numIter)) != 1) {
      stop("numIter is not an integer")
    }
    
    if(floor(numIter) != numIter){
      warning("numIter is not an integer, floor(numIter) will be used")
    }
    numIter = as.integer(numIter)
    
    
    if (numIter <= 0) {
      stop("numIter is not a positive integer")
    }
    
    # Get the # of rows and cols in X and checking against adversarial user
    
    n <- as.integer(nrow(X)) # number of rows in X
    stopifnot(!is.na(n))
    n_col_X = as.integer(ncol(X))
    stopifnot(!is.na(n_col_X))
    
    
    # Check that number of clusters are not greater than number of data points
    stopifnot(K <= n)
    
    # If number of clusters same as number of data, warn user
    if (K == n) {
      warning("Number of Clusters is the same as the number of data points")
    }
    
    # Check whether M is NULL or not. If NULL, initialize based on K random points from X.
    if (is.null(M)) {
      # Get random sample of the rows of X
      M <- X[sample(1:n, size = K), , drop = FALSE] 
      
    }
    # If not NULL, check for compatibility with X dimensions and K.
    if (!is.null(M)) {
      # Coerce M to matrix if it's a data.frame
      if(is.data.frame(M)){
        M = as.matrix(M)
      }
      
      stopifnot(exprs = {
        is.matrix(M)
        ! any(is.na(X))
        is.double(M)
        ncol(M) == n_col_X
        nrow(M) == K
        })
      
    }
  }
  
  
  
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X, K, M, numIter)
  
  # Return the class assignments
  return(Y)
}