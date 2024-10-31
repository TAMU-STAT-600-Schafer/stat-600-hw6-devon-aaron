#' K-means clustering for a data matrix X
#'
#' @param X A n x p matrix of numeric data
#' @param K The number of clusters. 
#' @param M (Optional) A K x p matrix of data used as initial centers. If this argument is used, number of rows in M must equal non-optional argument K
#' @param numIter The max number of iterations used in K-means clustering algorithm.
#'
#' @return A vector of length $n$ containing integers between 1 and K of cluster assignments.
#' @export
#'
#' @examples
#' 
#' ##### Example 1
#' set.seed(0)
#' 
#' n1 = 100
#' n2 = 100
#' p = 5
#' 
#' X = rbind(matrix(rnorm(n1*p, mean = -2), nrow = n1, ncol = p),
#'           matrix(rnorm(n2*p, mean = 5),  nrow = n2, ncol = p))
#' 
#' K = 2
#' 
#' cluster_assignment = MyKmeans(X, K)
#' 
#' 
#' #####################
#' ##### Example 2
#'
#' set.seed(0)
#' 
#' n1 = 50
#' n2 = 50
#' n3 = 50
#' n4 = 50
#' 
#' p = 2
#' 
#' X = rbind(matrix(rnorm(n1*p, mean = -3, sd = .1), nrow = n1, ncol = p),
#'           matrix(rnorm(n2*p, mean = -1, sd = .1), nrow = n2, ncol = p),
#'           matrix(rnorm(n3*p, mean =  1, sd = .1), nrow = n3, ncol = p),
#'           matrix(rnorm(n4*p, mean =  3, sd = .1), nrow = n4, ncol = p))
#' 
#' K = 4
#' 
#' M = X[sample.int(n1+n2+n3+n4, size = K, replace = TRUE), , drop = FALSE]
#' cluster_assignment = MyKmeans(X, K, M)
#'  
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  ############# Adversarial user checks ##################
  {
    # Coerce M to matrix if it's a data.frame
    if(is.data.frame(X)){
      X = as.matrix(X)
    }
    
    stopifnot(exprs = {
      # Check K and numIter are double single value
      is.numeric(K)
      is.numeric(numIter)
      length(as.integer(K)) == 1
      length(as.integer(numIter)) == 1
      !is.na(K)
      !is.na(numIter)
      
      # Check to make sure X makes sense
      is.matrix(X)
      !any(is.na(X))
      is.numeric(X)
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
  Y = MyKmeans_c(X, K, M, numIter) + 1
  
  # Return the class assignments
  return(Y)
}