// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Code to count unique elements in array is given by:
// https://www.geeksforgeeks.org/count-distinct-elements-in-an-array/
int count_unique(arma::uvec arr, int n){
  int out = 0;
  
  for(int i = 0; i < n; i++){
    int l = 0;
    for(int j = 0; j < i; j++){
      if(arr(i) == arr(j)){
        break;
      }
      l++;
    }
    
    if (i == l){
      out++;
    }

  }
  return(out);
} 


// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                            const arma::mat& M, int numIter = 100){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n); // to store cluster assignments
    arma::uvec Y_old(n);
    
    
    // ############# Implement K-means algorithm. ##############
    //  It should stop when either
    //  (i) the centroids don't change from one iteration to the next (exactly the same), or
    //  (ii) the maximal number of iterations was reached, or
    //  (iii) one of the clusters has disappeared after one of the iterations (in which case the error message is returned)
    
    
    // Initialize any additional parameters if needed
    arma::colvec rowSumsX2(n);
    arma::vec rowSumsM2(K);
    arma::mat crossprodXM_sumM2(n, p);
    arma::mat Xnew(n,p);
    Xnew = X;

    rowSumsX2 = arma::sum(arma::square(Xnew), 1);
    arma::mat M_centroids(K,p);
    
    M_centroids = M;
    
    // For loop with kmeans algorithm
    for(int iter = 0; iter < numIter; iter++){
      Y_old = Y;
      rowSumsM2 = arma::sum(arma::square(M_centroids), 1); 
      
      if(M.n_rows > 1){
        crossprodXM_sumM2 = -2.0 * (X * M_centroids.t());
        crossprodXM_sumM2.each_row() += rowSumsM2.t();
      }
      // 
      if(M.n_rows <= 1){
      // I want to return all 0's if this happens
        arma::uvec out(n);
        return(out);
      }
      
      Y = arma::index_min(crossprodXM_sumM2, 1);

      // Check if centroids have changed. Return cluster assignments if not.
      if(arma::all(Y == Y_old)){
        return(Y);
      }
      
      
      // Check if one of the clusters has disappeared, return an error if so.
      int num_unique_Y = count_unique(Y, n);
      
      
      if( num_unique_Y != K ){
        break;
      }
      
      for(int k = 0; k < K; k++){
        M_centroids.row(k) = arma::mean(X.rows(arma::find(Y == k)), 0);
        
      }
    }
    
    // Returns the vector of cluster assignments
    return(Y);
}

