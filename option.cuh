#pragma once

#include <cuda.h>
#include "greeks.cuh"

namespace qmc {

  template <class O>
  struct Option {
    __device__ virtual void SimulatePaths(const int N, float *d_z) = 0;
    __device__ virtual void CalculatePayoffs(Greeks<double> &greeks) = 0;
  };

  template <class O>
  struct ArithmeticAsian : Option<O> {
    O s1, s_tilde, avg_s1, s_max;
    O psi_d, payoff, delta, vega, gamma, theta;
    O vega_inner_sum;
    O lr_delta, lr_vega, lr_gamma, lr_theta;
    float z, z1, W1, W_tilde;
    int ind, ind_zero;

    void PrintName() {
      printf("\n**OPTION** : ArithmeticAsian\n");
    }

    __device__ void SimulatePaths(const int N, float *d_z) override { 
        //Algorithm 3
        // Initial setup
        z   = d_z[ind]; 
        z1 = z; // Capture z1 for lr_estimate

        // Initial path values
        W1 = sqrt(dt) * z;
        W_tilde = W1;
        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);
        
        // Set initial values required for greek estimates
        avg_s1 = s1;
        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));

        // Simulate over rest of N timesteps
        for (int n = 1; n < N; n++) { 
          ind += blockDim.x;      // shift pointer to random variable
          z = d_z[ind]; 

          // Stock dynamics
          W_tilde = W_tilde + sqrt(dt) * z; //Random variable 
          s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          // Required for greek estimations
          vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
          lr_vega += ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));
          avg_s1 += s1; 

        } 
        avg_s1 /= N; 
        vega_inner_sum /= N;
      } 

    __device__ void SimulatePathsQuasiBB(const int N, float *d_z, O *d_path) {
        int i;
        O a, b;
        ind_zero = ind;
        int h = N; // 2^m
        int m = static_cast<int>(log2f(h));

        //Algorithm 4
        d_path[ind_zero] = d_z[ind];

        for (int k = 1; k <= m; k++) { // k = 1,...,m
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; --j) {
            ind += blockDim.x;
            z = d_z[ind];
            a = O(0.5) * d_path[ind_zero + j * blockDim.x];
            b = sqrt(1.0 / (1 << (k+1)));
            d_path[ind_zero + i * blockDim.x] = a - b * z;
            i--;
            d_path[ind_zero + i * blockDim.x] = a + b * z;
            i--;
          }
        }
         
        W1 = d_path[ind_zero];
        W_tilde = W1;

        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);

        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        avg_s1 = s1;

        for (int k = 1; k < N; ++k) {
          W_tilde = W_tilde + d_path[ind_zero + k * blockDim.x];
          s_tilde = s0 * exp(omega * (dt*k - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*k - dt));
          avg_s1 += s1;
        }
        avg_s1 /= N;
        vega_inner_sum /= N;
      }

    __device__ void CalculatePayoffs(Greeks<double> &greeks) override {
        psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        payoff = exp(-r * T) * max(avg_s1 - k, O(0.0));

        // CPW Delta
        delta = exp(r * (dt - T)) * (avg_s1 / s0) 
          * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - T)) * (O(1.0) - normcdf(psi_d - sigma*sqrt(dt)))
          * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
          * N_PDF(psi_d);

        // CPW Theta
        theta = -exp(-r * T) * (avg_s1 / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

        // Likelihood ratio
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
          (z1 / (s0 * s0 * sigma * sqrt(dt))));
        lr_theta = -r*(lr_delta*s0+payoff);


        // Store results in respective arrays
        greeks.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        greeks.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        greeks.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega;
        greeks.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        greeks.theta[threadIdx.x + blockIdx.x*blockDim.x] = theta;
        greeks.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
        greeks.lr_vega[threadIdx.x + blockIdx.x*blockDim.x] = lr_vega;
        greeks.lr_gamma[threadIdx.x + blockIdx.x*blockDim.x] = lr_gamma;
        greeks.lr_theta[threadIdx.x + blockIdx.x*blockDim.x] = lr_theta;
      }

    __host__ void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt,
          float sigma, O s0, O k, float T, float omega, Greeks<double> &results) {
        ind = 0;

        for (int i = 0; i < NPATHS; ++i) {
          // Initial setup
          z   = h_z[ind]; 
          z1 = z; // Capture z1 for lr_estimate

          // Initial path values
          W1 = sqrt(dt) * z;
          W_tilde = W1;
          s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
          s1 = s0 * exp(omega * dt + sigma * W1);
          
          // Set initial values required for greek estimates
          avg_s1 = s1;
          vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
          lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));

          // Simulate over rest of N timesteps
          for (int n = 1; n < N; n++) { 
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
            avg_s1 += s1; 
          } 
          avg_s1 /= N; 
          vega_inner_sum /= N;

          psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

          payoff = exp(-r * T) * max(avg_s1 - k, O(0.0));

          delta = exp(r * (dt - T)) * (avg_s1 / s0) 
            * (O(1.0) - N_CDF(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T)) * (O(1.0) - N_CDF(psi_d - sigma*sqrt(dt)))
            * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
            * N_PDF(psi_d);
          theta = -exp(-r * T) * (avg_s1 / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
            (z1 / (s0 * s0 * sigma * sqrt(dt))));
          lr_theta = -r*(lr_delta*s0+payoff);

          results.price[i] = payoff;
          results.delta[i] = delta;
          results.vega[i] = vega;
          results.gamma[i] = gamma;
          results.theta[i] = theta;
          results.lr_delta[i] = lr_delta;
          results.lr_vega[i] = lr_vega;
          results.lr_gamma[i] = lr_gamma;
          results.lr_theta[i] = lr_theta;
        }
      }
  };

  template <class O>
  struct BinaryAsian : Option<O> {
    O s1, s_tilde, avg_s1, s_max;
    O psi_d, payoff, delta, vega, gamma, theta;
    O vega_inner_sum;
    O lr_delta, lr_vega, lr_gamma, lr_theta;
    float z, z1, W1, W_tilde;
    int ind, ind_zero;    

    void PrintName() {
      printf("\n **OPTION** : BinaryAsian\n");
    }

    __device__ void SimulatePaths(const int N, float *d_z) override {
        //Algorithm 3
        // Initial setup
        z   = d_z[ind]; 
        z1 = z; // Capture z1 for lr_estimate

        // Initial path values
        W1 = sqrt(dt) * z;
        W_tilde = W1;
        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);
        
        // Set initial values required for greek estimates
        avg_s1 = s1;
        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));

        // Simulate over rest of N timesteps
        for (int n = 1; n < N; n++) { 
          ind += blockDim.x;      // shift pointer to random variable
          z = d_z[ind]; 

          // Stock dynamics
          W_tilde = W_tilde + sqrt(dt) * z;
          s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          // Required for greek estimations
          vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
          lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
          avg_s1 += s1; 
        } 
        avg_s1 /= N;
        vega_inner_sum /= N;
      }

    __device__ void SimulatePathsQuasiBB(const int N, float *d_z, O *d_path) {
        int i;
        O a, b;
        ind_zero = ind;
        int h = N; // 2^m
        int m = static_cast<int>(log2f(h));
        //Algorithm 4
        d_path[ind_zero] = d_z[ind];
        for (int k = 1; k <= m; k++) { // k = 1,...,m
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; --j) {
            ind += blockDim.x;
            z = d_z[ind];
            a = O(0.5) * d_path[ind_zero + j * blockDim.x];
            b = sqrt(1.0 / (1 << (k+1)));
            d_path[ind_zero + i * blockDim.x] = a - b * z;
            i--;
            d_path[ind_zero + i * blockDim.x] = a + b * z;
            i--;
          }
        }
         
        W1 = d_path[ind_zero];
        W_tilde = W1;

        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);

        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        avg_s1 = s1;

        for (int k = 1; k < N; ++k) {
          W_tilde = W_tilde + d_path[ind_zero + k * blockDim.x];
          s_tilde = s0 * exp(omega * (dt*k - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*k - dt));
          avg_s1 += s1;
        }
        avg_s1 /= N;
        vega_inner_sum /= N;
      }

    __device__ void CalculatePayoffs(Greeks<double> &greeks) override {
        psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        if (avg_s1-k > O(0.0)){
          payoff=exp(-r * T);
        }
        else{
          payoff = O(0.0);
        }

        // CPW Delta
        delta = (exp(-r * T) / (s0 * sigma * sqrt(dt))) * N_PDF(psi_d);

        // CPW Vega
        vega = exp(-r * T) * N_PDF(psi_d) *
          ((O(1.0) / (sigma * sqrt(dt) * avg_s1)) * vega_inner_sum +
           psi_d / sigma - sqrt(dt));

        // CPW Gamma
        gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d)
          * ((psi_d / (sigma * sqrt(dt)) - O(1.0)));

        //CPW Theta 
        theta = -r * exp(-r * T) * ((avg_s1 > k) ? 1.0 : 0.0);

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
          (z1 / (s0 * s0 * sigma * sqrt(dt))));
        lr_theta = -lr_delta * r;

        
        // Store results in respective arrays
        greeks.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        greeks.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        greeks.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega;
        greeks.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        greeks.theta[threadIdx.x + blockIdx.x*blockDim.x] = theta;
        greeks.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
        greeks.lr_vega[threadIdx.x + blockIdx.x*blockDim.x] = lr_vega;
        greeks.lr_gamma[threadIdx.x + blockIdx.x*blockDim.x] = lr_gamma;
        greeks.lr_theta[threadIdx.x + blockIdx.x*blockDim.x] = lr_theta;
      }

    __host__ void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt,
          float sigma, O s0, O k, float T, float omega, Greeks<double> &results) {
        ind = 0;

        for (int i = 0; i < NPATHS; ++i) {
          // Initial setup
          z   = h_z[ind]; 
          z1 = z; // Capture z1 for lr_estimate

          // Initial path values
          W1 = sqrt(dt) * z;
          W_tilde = W1;
          s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
          s1 = s0 * exp(omega * dt + sigma * W1);
          
          // Set initial values required for greek estimates
          avg_s1 = s1;
          vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
          lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));
          
          // Simulate over rest of N timesteps
          for (int n = 1; n < N; n++) { 
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
            avg_s1 += s1; 
          } 
          avg_s1 /= N; 
          vega_inner_sum /= N;

          psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

          payoff = avg_s1 - k > O(0.0) ? exp(-r * T) : O(0.0);

          delta = (exp(-r * T) / (s0 * sigma * sqrt(dt))) * N_PDF(psi_d);
          
          vega = exp(-r * T) * N_PDF(psi_d) *
            ((O(1.0) / (sigma * sqrt(dt) * avg_s1)) * vega_inner_sum +
             psi_d / sigma - sqrt(dt));

          gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d)
          * ((psi_d / (sigma * sqrt(dt)) - O(1.0)));

          theta = -r * exp(-r * T) * ((avg_s1 > k) ? 1.0 : 0.0);


          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
            (z1 / (s0 * s0 * sigma * sqrt(dt))));
          lr_theta = -lr_delta * r;

          results.price[i] = payoff;
          results.delta[i] = delta;
          results.vega[i] = vega;
          results.gamma[i] = gamma;
          results.theta[i] = theta;
          results.lr_delta[i] = lr_delta;
          results.lr_vega[i] = lr_vega;
          results.lr_gamma[i] = lr_gamma;
          results.lr_theta[i] = lr_theta;
        }
      }
  };

  template <class O>
  struct Lookback : Option<O> {
    O s1, s_tilde, s_max;
    O psi_d, payoff, delta, vega, gamma, theta;
    O vega_inner_sum;
    O lr_delta, lr_vega, lr_gamma, lr_theta;
    float z, z1, W1, W_tilde;
    int ind, ind_zero;

    void PrintName() {
      printf("\n**OPTION** : Lookback\n");
    }
    
    __device__ void SimulatePaths(const int N, float *d_z) override {
        // Initial setup
        z   = d_z[ind]; 
        z1 = z; // Capture z1 for lr_estimate

        // Initial path values
        W1 = sqrt(dt) * z;
        W_tilde = W1;
        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);

        // Set initial values required for greek estimates
        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));
        s_max = s1;

        for (int n = 1; n < N; n++) {
          ind += blockDim.x;      // shift pointer to random variable
          z = d_z[ind]; 

          // Stock dynamics
          W_tilde = W_tilde + sqrt(dt) * z;
          s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          // Required for greek estimations
          if (s1 > s_max) {
            s_max = s1;
            vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
          } 
          lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
        }
      }

    __device__ void SimulatePathsQuasiBB(const int N, float *d_z, O *d_path) {
        int i;
        O a, b;
        ind_zero = ind;
        int h = N; // 2^m
        int m = static_cast<int>(log2f(h));

        d_path[ind_zero] = d_z[ind];

        for (int k = 1; k <= m; k++) { // k = 1,...,m
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; --j) {
            ind += blockDim.x;
            z = d_z[ind];
            a = O(0.5) * d_path[ind_zero + j * blockDim.x];
            b = sqrt(1.0 / (1 << (k+1)));
            d_path[ind_zero + i * blockDim.x] = a - b * z;
            i--;
            d_path[ind_zero + i * blockDim.x] = a + b * z;
            i--;
          }
        }
         
        W1 = d_path[ind_zero];
        W_tilde = W1;

        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);

        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        s_max = s1;

        for (int k = 1; k < N; ++k) {
          W_tilde = W_tilde + d_path[ind_zero + k * blockDim.x];
          s_tilde = s0 * exp(omega * (dt*k - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          if (s1 > s_max) {
            s_max = s1;
            vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*k - dt));
          }
        }
      }

    __device__ void CalculatePayoffs(Greeks<double> &greeks) override {
        psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        payoff = exp(-r * T) * max(s_max - k, O(0.0));

        // CPW Delta
        delta = exp(r * (dt - T)) * (s_max / s0)
          * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - T)) * (O(1.0) - normcdf(psi_d - sigma*sqrt(dt)))
          * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
          * N_PDF(psi_d);
        
        //CPW Theta
        theta = -exp(-r * T) * (s_max / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
          (z1 / (s0 * s0 * sigma * sqrt(dt))));
        lr_theta = 0; //TODO
        
        // Store results in respective arrays
        greeks.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        greeks.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        greeks.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega;
        greeks.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        greeks.theta[threadIdx.x + blockIdx.x*blockDim.x] = theta;
        greeks.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
        greeks.lr_vega[threadIdx.x + blockIdx.x*blockDim.x] = lr_vega;
        greeks.lr_gamma[threadIdx.x + blockIdx.x*blockDim.x] = lr_gamma;
        greeks.lr_theta[threadIdx.x + blockIdx.x*blockDim.x] = lr_theta;
      }

    __host__ void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt,
          float sigma, O s0, O k, float T, float omega, Greeks<double> &results) {
        ind = 0;
        for (int i = 0; i < NPATHS; ++i) {
          // Initial setup
          z   = h_z[ind]; 
          z1 = z; 

          // Initial path values
          W1 = sqrt(dt) * z;
          W_tilde = W1;
          s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
          s1 = s0 * exp(omega * dt + sigma * W1);
          
          // Set initial values required for greek estimates
          vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
          lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));
          s_max = s1;

          for (int n=0; n<N; n++) {
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            if (s1 > s_max) {
              s_max = s1;
              vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*n - dt)); 
            } 
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
          }

          psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt));

          payoff = exp(-r * T) * max(s_max - k, 0.0f);

          delta = exp(r * (dt - T)) * (s_max / s0)
            * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T)) * (O(1.0) - N_CDF(psi_d - sigma*sqrt(dt)))
            * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
            * N_PDF(psi_d);

          theta = -exp(-r * T) * (s_max / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
            (z1 / (s0 * s0 * sigma * sqrt(dt))));
          lr_theta = 0; //TODO

          results.price[i] = payoff;
          results.delta[i] = delta;
          results.vega[i] = vega;
          results.gamma[i] = gamma;
          results.theta[i]=theta;
          results.lr_delta[i] = lr_delta;
          results.lr_vega[i] = lr_vega;
          results.lr_gamma[i] = lr_gamma;
          results.lr_theta[i]=lr_theta;
        }
      }
    };
  template <class O>
  struct ForwardStartEuropeanCall : Option<O> {
    O s1, s_tilde, avg_s1, k_min,s_max;
    O psi_d, payoff, delta, vega, gamma, theta;
    O vega_inner_sum;
    O lr_delta, lr_vega, lr_gamma, lr_theta;
    float z, z1, W1, W_tilde;
    int ind, ind_zero; 

    void PrintName() {
      printf("\n**OPTION** : ForwardStartEuropeanCall\n");
    }

     __device__ void SimulatePaths(const int N, float *d_z) override {
        // Initial setup
        z   = d_z[ind]; 
        z1 = z; // Capture z1 for lr_estimate

        // Initial path values
        W1 = sqrt(dt) * z;
        W_tilde = W1;
        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);

        // Set initial values required for greek estimates
        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));
        k_min = s1;

        for (int n = 1; n < N; n++) {
          ind += blockDim.x;      // shift pointer to random variable
          z = d_z[ind]; 

          // Stock dynamics
          W_tilde = W_tilde + sqrt(dt) * z;
          s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          // Required for greek estimations
          if (s1 < k_min && n<N/10) {
            k_min = s1;
            vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
          } 
          lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
        }
        s_max=0;
      }

    __device__ void SimulatePathsQuasiBB(const int N, float *d_z, O *d_path) {
        int i;
        O a, b;
        ind_zero = ind;
        int h = N; // 2^m
        int m = static_cast<int>(log2f(h));

        d_path[ind_zero] = d_z[ind];

        for (int k = 1; k <= m; k++) { // k = 1,...,m
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; --j) {
            ind += blockDim.x;
            z = d_z[ind];
            a = O(0.5) * d_path[ind_zero + j * blockDim.x];
            b = sqrt(1.0 / (1 << (k+1)));
            d_path[ind_zero + i * blockDim.x] = a - b * z;
            i--;
            d_path[ind_zero + i * blockDim.x] = a + b * z;
            i--;
          }
        }
         
        W1 = d_path[ind_zero];
        W_tilde = W1;

        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);

        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        k_min = s1;

        for (int k = 1; k < N; ++k) {
          W_tilde = W_tilde + d_path[ind_zero + k * blockDim.x];
          s_tilde = s0 * exp(omega * (dt*k - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          if (s1 <k_min && k<N/10) {
            k_min = s1;
            vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*k - dt));
          }
        }
        s_max=0;
      }
    __device__ void CalculatePayoffs(Greeks<double> &greeks) override {
        psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        payoff = exp(-r * T) * max(s_max - k, O(0.0));

        // CPW Delta
        delta = exp(r * (dt - T)) * (s_max / s0)
          * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - T)) * (O(1.0) - normcdf(psi_d - sigma*sqrt(dt)))
          * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
          * N_PDF(psi_d);
        
        //CPW Theta
        theta = -exp(-r * T) * (s_max / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
          (z1 / (s0 * s0 * sigma * sqrt(dt))));
        lr_theta = 0; //TODO
        
        // Store results in respective arrays
        greeks.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        greeks.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        greeks.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega;
        greeks.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        greeks.theta[threadIdx.x + blockIdx.x*blockDim.x] = theta;
        greeks.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
        greeks.lr_vega[threadIdx.x + blockIdx.x*blockDim.x] = lr_vega;
        greeks.lr_gamma[threadIdx.x + blockIdx.x*blockDim.x] = lr_gamma;
        greeks.lr_theta[threadIdx.x + blockIdx.x*blockDim.x] = lr_theta;
      }

    __host__ void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt,
          float sigma, O s0, O k, float T, float omega, Greeks<double> &results) {
        ind = 0;
        for (int i = 0; i < NPATHS; ++i) {
          // Initial setup
          z   = h_z[ind]; 
          z1 = z; 

          // Initial path values
          W1 = sqrt(dt) * z;
          W_tilde = W1;
          s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
          s1 = s0 * exp(omega * dt + sigma * W1);
          
          // Set initial values required for greek estimates
          vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
          lr_vega = ((z*z - O(1.0)) / sigma) - (z * sqrt(dt));
          s_max = s1;

          for (int n=0; n<N; n++) {
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            if (s1 > s_max) {
              s_max = s1;
              vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*n - dt)); 
            } 
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
          }

          psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt));

          payoff = exp(-r * T) * max(s_max - k, 0.0f);

          delta = exp(r * (dt - T)) * (s_max / s0)
            * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T)) * (O(1.0) - N_CDF(psi_d - sigma*sqrt(dt)))
            * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
            * N_PDF(psi_d);

          theta = -exp(-r * T) * (s_max / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
            (z1 / (s0 * s0 * sigma * sqrt(dt))));
          lr_theta = 0; //TODO

          results.price[i] = payoff;
          results.delta[i] = delta;
          results.vega[i] = vega;
          results.gamma[i] = gamma;
          results.theta[i]=theta;
          results.lr_delta[i] = lr_delta;
          results.lr_vega[i] = lr_vega;
          results.lr_gamma[i] = lr_gamma;
          results.lr_theta[i]=lr_theta;
        }
      }
  };

}
