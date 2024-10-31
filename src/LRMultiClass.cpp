// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// One Hot Encode Matrices:
arma::mat uvec_one_hot(arma::uvec y, int n, int K) {
  
  // Initialize zero matrix
  arma::mat one_hot_mat = arma::zeros<arma::mat>(n, K);
  
  for(int i = 0; i < n; i++){
    
    int col_index = y[i];
    
    one_hot_mat(i, col_index) = 1.0;
    
  }
  
  return one_hot_mat;
  
}




// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// // beta_init - p x K matrix of starting beta values (always supplied in right format)

// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct

    ////////////////////////////////
    // Initialize some parameters //
    ////////////////////////////////

    int K = max(y) + 1; // number of classes
    int p = X.n_cols;
    int n = X.n_rows;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values

    ////////////////////////////////////////////////
    // Initialize anything else that you may need //
    ////////////////////////////////////////////////

    // ////
    // Calculate corresponding pk, objective value f(beta_init) given the starting point beta_init
    // ////

    // Compute pk: //
    // ////
    // Numerator:
    arma::mat Xb = X * beta;
    arma::mat exp_Xb = arma::exp(Xb);
    // Denominator:
    arma::vec sum_exp_Xb = arma::sum(exp_Xb, 1); 
    // pk:
    arma::mat p_k = exp_Xb.each_col() / sum_exp_Xb;
    
    
    // Compute Objective Value f(beta_init): //
    // ////
    // Negative Log Likelihood:
    arma::mat y_indicator = uvec_one_hot(y, n, K); // One-hot encode y uvec

    
    double obj = -arma::accu(y_indicator % arma::log(p_k)) + (lambda / 2.0) * arma::accu(arma::square(beta));
    
    // Append Objective value f(beta_init) to Main Objective Vector:
    objective[0] = obj;

    // ////
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    // ////
    // Initialize Terms:
    arma::mat X_tran = X.t();
    arma::mat lambda_I = lambda * arma::eye<arma::mat>(p, p);
    arma::vec g(K);
    arma::vec W(n);
    arma::mat h(K, K);
    arma::mat X_W(n,p);

    
    // Loop: Within one iteration perform the update, calculate updated objective function 
    // ////
    for(int i = 0; i < numIter; i++){

      // Newly added for W computation:

      for(int k = 0; k < K; k++){

        // W term configuration (in Hessian)
        //  W in this case is the kth diagonal element of main W mat
        W = p_k.col(k) % (1 - p_k.col(k));
        
        // weighted multiplication of matrix X
        X_W = X.each_col() % arma::sqrt(W); 
        
        // grandient Update:
        g = X_tran * (p_k.col(k) - y_indicator.col(k)) + (lambda * beta.col(k));
        
        // Hessian Update:
        h = (X_W.t() * X_W) + (lambda_I); 
        
        // Damped Newton's Update:
        beta.col(k) -= (eta * (arma::solve(h, g))); 

      }
      
      // Recalculate probabilities in p_k
      Xb = X * beta;
      exp_Xb = arma::exp(Xb);
      // Rowsums:
      sum_exp_Xb = arma::sum(exp_Xb, 1);
      // pk:
      p_k = exp_Xb.each_col() / sum_exp_Xb;
      
      
      obj = -arma::accu(y_indicator % arma::log(p_k)) + (lambda / 2.0) * arma::accu(arma::square(beta));
      // Append Objective value f(beta_init) to Main Objective Vector:
      objective[i + 1] = obj;

    }
    
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
