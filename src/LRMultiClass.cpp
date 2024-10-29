// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
// EDIT: Line below required to remove Rcpp errors
// #include "Rcpp"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]

// ////
// FUNCTIONS
// ////
// One Hot Encode Matrices:
arma::mat uvec_one_hot(arma::uvec y, int n, int K) {
  
  // Initialize zero matrix
  arma::mat one_hot_mat = arma::zeros<arma::mat>(n, K);
  
  for(int i = 0; i < n; i++){
    
    int col_index = y[i] - 1;
    
    one_hot_mat(i, col_index) = 1;
    
  }
  
  return one_hot_mat;
  
}

// [[Rcpp::export]]

// Sum Diagonals of Matrices:
double sum_diag(const arma::mat& main_mat, int num_col){
  
  double diagonal_sum = 0.0;
  
  for(int i = 0; i < num_col; i++){
    
    diagonal_sum = diagonal_sum + main_mat(i, i);
    
  }
  
  return diagonal_sum;
  
}

// ////
// ////

// [[Rcpp::export]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// // beta_init - p x K matrix of starting beta values (always supplied in right format)
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
    arma::mat Xb = X * beta_init;
    arma::mat exp_Xb = arma::exp(Xb);
    // Denominator:
    arma::colvec sum_exp_Xb = arma::sum(exp_Xb, 1); // Essentially rowsum()
    // pk:
    arma::mat p_k = exp_Xb / sum_exp_Xb;

    // Compute Objective Value f(beta_init): //
    // ////
    // Negative Log Likelihood:
    arma::mat y_indicator = uvec_one_hot(y, n, K); // One-hot encode y uvec
    arma::mat objective_obj1_mat = y_indicator * arma::log(p_k).t(); // Compute matrix in Negative Log Likelihood derivation
    int num_col_obj1_mat = objective_obj1_mat.n_cols;
    double objective_obj1 = (-1) * sum_diag(objective_obj1_mat, num_col_obj1_mat); // Negative Log Likelihood
    // Ridge Penalty:
    double ridge_pen = (lambda / 2) * (arma::accu(arma::sum(arma::square(beta_init), 0)));

    // Objective Value f(beta_init):
    double objective_obj = objective_obj1 + ridge_pen;

    // Append Objective value f(beta_init) to Main Objective Vector:
    objective[0] = objective_obj;


    // ////
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    // ////
    // Initialize Terms:
    arma::mat X_tran = X.t();
    arma::mat Identity = arma::eye<arma::mat>(p, p);
    
    // Loop: Within one iteration perform the update, calculate updated objective function and training/testing errors in %
    // ////
    for(int i = 0; i < numIter; i++){
      
      
      for(int k = 0; k < K; k++){
        
        // W term configuration (in Hessian)
        //  W in this case is the kth diagonal element of main W mat
        arma::mat W = p_k.col(k) % (1 - p_k.col(k));
        
        // Generating Function Update:
        arma::mat g = X_tran * (p_k.col(k) - y_indicator.col(k)) + (lambda * beta.col(k));
        // Hessian Update:
        arma::mat h = ((X % W) * X) + (lambda * Identity);
        // Damped Newton's Update:
        beta.col(k) = beta.col(k) - (eta * (h.i() * g));
        
      }
      
      // Update pk
      // Num:
      arma::mat Xb = X * beta;
      arma::mat exp_Xb = arma::exp(Xb);
      // Denom:
      arma::colvec sum_exp_Xb = arma::sum(exp_Xb, 1);
      // pk:
      arma::mat p_k = exp_Xb / sum_exp_Xb;
      
      // Compute Objective Value f(beta):
      arma::mat log_pk = arma::log(p_k);
      double neg_log_lik = -(arma::accu(y_indicator * log_pk)); // Negative Log Likelihood
      double ridge_reg = (lambda / 2) * (arma::accu(arma::sum(arma::square(beta), 0))); // Ridge Penalty
      double objective_obj = neg_log_lik + ridge_reg; // Objective Value
      
      objective[i + 1] = objective_obj; // Append
      
    }
    
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}






