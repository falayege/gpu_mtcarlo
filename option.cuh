#pragma once

#include <cuda.h>
namespace qmc {

  __host__ __device__ float N_PDF(float x) {
    return exp(-0.5f * x * x) * (1.0f / sqrt(2.0f * M_PI));
  }

  __host__ float N_CDF(float x) {
    return std::erfc(-x/std::sqrt(2))/2;
  }



  enum Method {
    STANDARD, QUASI, QUASI_BB
  };

  __constant__ int   N, PATHS;
  __constant__ float T, r, sigma, dt, omega, s0, k;

  template <class T>
  struct LikelihoodRatios {
    T delta = 0, vega = 0, gamma = 0, theta = 0;
    T err_delta = 0, err_vega = 0, err_gamma = 0, err_theta= 0; 
  };

  template <class T>
  struct Greeks {
    T *price, *delta, *vega, *gamma, *theta;
    T *lr_delta, *lr_vega, *lr_gamma, *lr_theta;
    double avg_price = 0.0, avg_delta = 0.0, avg_vega = 0.0, avg_gamma = 0.0, avg_theta = 0.0;
    double err_price = 0.0, err_delta = 0.0, err_vega = 0.0, err_gamma = 0.0, err_theta=0.0;
    double avg_lr_delta = 0.0, avg_lr_vega = 0.0, avg_lr_gamma = 0.0, avg_lr_theta =0.0;
    double err_lr_delta = 0.0, err_lr_vega = 0.0, err_lr_gamma = 0.0, err_lr_theta=0.0;

    __host__ void ClearHost(const int size) {
        avg_price = 0.0; avg_delta = 0.0; avg_vega = 0.0; avg_gamma = 0.0; avg_theta = 0.0;
        err_price = 0.0; err_delta = 0.0; err_vega = 0.0; err_gamma = 0.0; err_theta = 0.0;
        avg_lr_delta = 0.0; avg_lr_vega = 0.0; avg_lr_gamma = 0.0; avg_lr_theta = 0.0;
        err_lr_delta = 0.0; err_lr_vega = 0.0; err_lr_gamma = 0.0; err_lr_theta = 0.0;
      }

    __host__ void ClearDevice(const int size) {
        avg_price = 0.0; avg_delta = 0.0; avg_vega = 0.0; avg_gamma = 0.0; avg_theta = 0.0;
        err_price = 0.0; err_delta = 0.0; err_vega = 0.0; err_gamma = 0.0; err_theta = 0.0;
        avg_lr_delta = 0.0; avg_lr_vega = 0.0; avg_lr_gamma = 0.0; avg_lr_theta = 0.0;
        err_lr_delta = 0.0; err_lr_vega = 0.0; err_lr_gamma = 0.0; err_lr_theta = 0.0;
      }

    __host__ void AllocateHost(const int size) {
        price = (T *) malloc(sizeof(T) * size);
        delta = (T *) malloc(sizeof(T) * size);
        vega = (T *) malloc(sizeof(T) * size);
        gamma = (T *) malloc(sizeof(T) * size);
        theta = (T *) malloc(sizeof(T) * size);
        lr_delta = (T *) malloc(sizeof(T) * size);
        lr_vega = (T *) malloc(sizeof(T) * size);
        lr_gamma = (T *) malloc(sizeof(T) * size);
        lr_theta = (T *) malloc(sizeof(T) * size);
      }

    __host__ void AllocateDevice(const int size) {
        cudaMalloc((void **) &price, sizeof(T) * size);
        cudaMalloc((void **) &delta, sizeof(T) * size);
        cudaMalloc((void **) &vega, sizeof(T) * size);
        cudaMalloc((void **) &gamma, sizeof(T) * size);
        cudaMalloc((void **) &theta, sizeof(T) * size);
        cudaMalloc((void **) &lr_delta, sizeof(T) * size);
        cudaMalloc((void **) &lr_vega, sizeof(T) * size);
        cudaMalloc((void **) &lr_gamma, sizeof(T) * size);
        cudaMalloc((void **) &lr_theta, sizeof(T) * size);
      }

    __host__ void CopyFromDevice(const int size, const Greeks<T> &device_greeks) {
        cudaMemcpy(price, device_greeks.price, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(delta, device_greeks.delta, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(vega, device_greeks.vega, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(gamma, device_greeks.gamma, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(theta, device_greeks.theta, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_delta, device_greeks.lr_delta, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_vega, device_greeks.lr_vega, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_gamma, device_greeks.lr_gamma, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_theta, device_greeks.lr_theta, sizeof(T) * size, cudaMemcpyDeviceToHost) ;
      }

    __host__ void ReleaseHost() {
        free(price);
        free(delta);
        free(vega);
        free(gamma);
        free(theta);
        free(lr_delta);
        free(lr_vega);
        free(lr_gamma);
        free(lr_theta);
      }

    __host__ void ReleaseDevice() {
        cudaFree(price);
        cudaFree(delta);
        cudaFree(vega);
        cudaFree(gamma);
        cudaFree(theta);
        cudaFree(lr_delta);
        cudaFree(lr_vega);
        cudaFree(lr_gamma);
        cudaFree(lr_theta);
      }

    __host__ void CalculateGreeks(const int size) {
        double price_sum = 0.0, price_sum2 = 0.0, delta_sum = 0.0, delta_sum2 = 0.0, vega_sum = 0.0, vega_sum2 = 0.0,
         gamma_sum = 0.0, gamma_sum2 = 0.0, theta_sum = 0.0, theta_sum2 = 0.0;
        double lr_delta_sum = 0.0, lr_delta_sum2 = 0.0, lr_vega_sum = 0.0, lr_vega_sum2 = 0.0,
         lr_gamma_sum = 0.0, lr_gamma_sum2 = 0.0, lr_theta_sum = 0.0, lr_theta_sum2 = 0.0;

        for (int i = 0; i < size; i++) {
          price_sum += price[i];
          price_sum2 += price[i] * price[i];
          delta_sum += delta[i];
          delta_sum2 += delta[i] * delta[i];
          vega_sum += vega[i];
          vega_sum2 += vega[i] * vega[i];
          gamma_sum += gamma[i];
          gamma_sum2 += gamma[i] * gamma[i];
          theta_sum += theta[i];
          theta_sum2 += theta[i] * theta[i];
          lr_delta_sum += lr_delta[i];
          lr_delta_sum2 += lr_delta[i] * lr_delta[i];
          lr_vega_sum += lr_vega[i];
          lr_vega_sum2 += lr_vega[i] * lr_vega[i];
          lr_gamma_sum += lr_gamma[i];
          lr_gamma_sum2 += lr_gamma[i] * lr_gamma[i];
          lr_theta_sum += lr_theta[i];
          lr_theta_sum2 += lr_theta[i] * lr_theta[i];

        }

        avg_price = price_sum / size;
        avg_delta = delta_sum / size;
        avg_vega = vega_sum / size;
        avg_gamma = gamma_sum / size;
        avg_theta = theta_sum / size;
        avg_lr_delta = lr_delta_sum / size;
        avg_lr_vega = lr_vega_sum / size;
        avg_lr_gamma = lr_gamma_sum / size;
        avg_lr_theta = lr_theta_sum/size;

        err_price = sqrt((price_sum2 / size - (price_sum / size) * (price_sum / size)) / size);
        err_delta = sqrt((delta_sum2 / size - (delta_sum / size) * (delta_sum / size)) / size);
        err_vega = sqrt((vega_sum2 / size - (vega_sum / size) * (vega_sum / size)) / size);
        err_gamma = sqrt((gamma_sum2 / size - (gamma_sum / size) * (gamma_sum / size)) / size);
        err_theta = sqrt((theta_sum2 / size - (theta_sum / size) * (theta_sum / size)) / size);
        err_lr_delta = sqrt((lr_delta_sum2 / size - (lr_delta_sum / size) * (lr_delta_sum / size)) / size);
        err_lr_vega = sqrt((lr_vega_sum2 / size - (lr_vega_sum / size) * (lr_vega_sum / size)) / size);
        err_lr_gamma = sqrt((lr_gamma_sum2 / size - (lr_gamma_sum / size) * (lr_gamma_sum / size)) / size);
        err_lr_theta = sqrt((lr_theta_sum2 / size - (lr_theta_sum / size) * (lr_theta_sum / size)) / size);
      }

    __host__ void PrintGreeks(bool print_header, const char* dev) {
        if (print_header) {
            printf("Greeks on %s:\n", dev);
        }

        printf("Price: %10.5f (±%12.5f)\n", avg_price, err_price);
        printf("Delta: %10.5f (±%12.5f)\n", avg_delta, err_delta);
        printf("Vega: %10.5f (±%12.5f)\n", avg_vega, err_vega);
        printf("Gamma: %10.5f (±%12.5f)\n", avg_gamma, err_gamma);
        printf("Theta: %10.5f (±%12.5f)\n", avg_theta, err_theta);
    }
  };


  template <class O>
  struct Option {
    __device__ virtual void SimulatePaths(const int N, float *d_z) = 0;
    __device__ virtual void CalculatePayoffs(Greeks<double> &greeks) = 0;
  };

  template <class O>
  struct ArithmeticAsian : Option<O> {
    O s1, s_tilde, avg_s1;
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
        //Algorithm 4
        int i;
        O a, b;
        ind_zero = ind;
        int h = N;
        int m = static_cast<int>(log2f(h));

        d_path[ind_zero] = d_z[ind];

        for (int k = 1; k <= m; k++) { 
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; j--) {
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

        for (int k = 1; k < N; k++) {
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
        delta = exp(r * (dt - T)) * (avg_s1 / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - T)) * (O(1.0) - normcdf(psi_d - sigma*sqrt(dt))) * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d);

        // CPW Theta
        theta = -exp(-r * T) * (avg_s1 / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

        // Likelihood ratio
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - (z1 / (s0 * s0 * sigma * sqrt(dt))));
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

        for (int i = 0; i < NPATHS; i++) {
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

          delta = exp(r * (dt - T)) * (avg_s1 / s0)  * (O(1.0) - N_CDF(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T)) * (O(1.0) - N_CDF(psi_d - sigma*sqrt(dt))) * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d);
          theta = -exp(-r * T) * (avg_s1 / s0) * (O(1.0) - N_CDF(psi_d - sigma * sqrt(dt))) * r;

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - (z1 / (s0 * s0 * sigma * sqrt(dt))));
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
    O s1, s_tilde, avg_s1;
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
        //Algorithm 4
        int i;
        O a, b;
        ind_zero = ind;
        int h = N;
        int m = static_cast<int>(log2f(h));
        //Algorithm 4
        d_path[ind_zero] = d_z[ind];
        for (int k = 1; k <= m; k++) { 
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; j--) {
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

        for (int k = 1; k < N; k++) {
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
        vega = exp(-r * T) * N_PDF(psi_d) *((O(1.0) / (sigma * sqrt(dt) * avg_s1)) * vega_inner_sum + psi_d / sigma - sqrt(dt));

        // CPW Gamma
        gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d)* ((psi_d / (sigma * sqrt(dt)) - O(1.0)));

        //CPW Theta 
        theta = -r * exp(-r * T) * ((avg_s1 > k) ? 1.0 : 0.0);

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - (z1 / (s0 * s0 * sigma * sqrt(dt))));
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

        for (int i = 0; i < NPATHS; i++) {
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
          
          vega = exp(-r * T) * N_PDF(psi_d) * ((O(1.0) / (sigma * sqrt(dt) * avg_s1)) * vega_inner_sum + psi_d / sigma - sqrt(dt));

          gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d) * ((psi_d / (sigma * sqrt(dt)) - O(1.0)));

          theta = -r * exp(-r * T) * ((avg_s1 > k) ? 1.0 : 0.0);


          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - (z1 / (s0 * s0 * sigma * sqrt(dt))));
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
        //Algorithm 4
        int i;
        O a, b;
        ind_zero = ind;
        int h = N; 
        int m = static_cast<int>(log2f(h));

        d_path[ind_zero] = d_z[ind];

        for (int k = 1; k <= m; k++) { 
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; j--) {
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

        for (int k = 1; k < N; k++) {
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
        delta = exp(r * (dt - T)) * (s_max / s0) * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - T)) * (O(1.0) - normcdf(psi_d - sigma*sqrt(dt)))* vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d);
        
        //CPW Theta
        theta = -exp(-r * T) * (s_max / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - (z1 / (s0 * s0 * sigma * sqrt(dt))));
        lr_theta = 0;
        
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
        for (int i = 0; i < NPATHS; i++) {
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

          delta = exp(r * (dt - T)) * (s_max / s0)* (1.0f - N_CDF(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T)) * (O(1.0) - N_CDF(psi_d - sigma*sqrt(dt))) * vega_inner_sum + k * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d);

          theta = -exp(-r * T) * (s_max / s0) * (O(1.0) - N_CDF(psi_d - sigma * sqrt(dt))) * r;

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) -  (z1 / (s0 * s0 * sigma * sqrt(dt))));
          lr_theta = 0; 

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
    O s1, s_tilde, avg_s1,k_min;
    O psi_d, payoff, delta, vega, gamma, theta;
    O vega_inner_sum;
    O lr_delta, lr_vega, lr_gamma, lr_theta;
    float z, z1, W1, W_tilde;
    int ind, ind_zero; 

    void PrintName() {
      printf("\n**OPTION** : ForwardStartEuropeanCall\n");
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
      }

    __device__ void SimulatePathsQuasiBB(const int N, float *d_z, O *d_path) {
        //Algorithm 4
        int i;
        O a, b;
        ind_zero = ind;
        int h = N;
        int m = static_cast<int>(log2f(h));

        d_path[ind_zero] = d_z[ind];

        for (int k = 1; k <= m; k++) { 
          i = (1 << k) - 1;
          for (int j = (1 << (k-1)) - 1; j >= 0; j--) {
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

        for (int k = 1; k < N; k++) {
          W_tilde = W_tilde + d_path[ind_zero + k * blockDim.x];
          s_tilde = s0 * exp(omega * (dt*k - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          if (s1 <k_min && k<N/10) {
            k_min = s1;
            vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*k - dt));
          }
        }
      }

__device__ void CalculatePayoffs(Greeks<double> &greeks) override {
        psi_d = (log(k_min) - log(s1) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        payoff = exp(-r * T*0.9) * max(s1 - k_min, O(0.0));

        // CPW Delta
        delta = exp(r * (dt - 0.9*T)) * (s1 / s0)* (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - 0.9*T)) * (O(1.0) - normcdf(psi_d - sigma*sqrt(dt))) * vega_inner_sum + k_min * exp(-r * 0.9*T) * N_PDF(psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * 0.9*T)) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d);
        
        //CPW Theta
        theta = -exp(-r * 0.9*T) * (s1 / s0) * (O(1.0) - normcdf(psi_d - sigma * sqrt(dt))) * r;

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) - (z1 / (s0 * s0 * sigma * sqrt(dt))));
        lr_theta = -payoff * r * exp(-r * (T*0.9));
        
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
        for (int i = 0; i < NPATHS; i++) {
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
          k_min = s1;

          for (int n=0; n<N; n++) {
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            if (s1 <k_min && n<N/10) {
              k_min = s1;
              vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*n - dt)); 
            } 
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
          }

          psi_d = (log(k_min) - log(s1) - omega * dt) / (sigma * sqrt(dt));

          payoff = exp(-r * T*0.9) * max(s1- k_min, 0.0f);

          delta = exp(r * (dt - T*0.9)) * (s1 / s0) * (1.0f - N_CDF(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T*0.9)) * (O(1.0) - N_CDF(psi_d - sigma*sqrt(dt))) * vega_inner_sum + k_min * exp(-r * T) * N_PDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T*0.9)) / (s0 * s0 * sigma * sqrt(dt))) * N_PDF(psi_d);

          theta = -exp(-r * T*0.9) * (s1 / s0) * (O(1.0) - N_CDF(psi_d - sigma * sqrt(dt))) * r;

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - O(1.0)) / (s0 * s0 * sigma * sigma * dt)) -  (z1 / (s0 * s0 * sigma * sqrt(dt))));
          lr_theta = -payoff * r * exp(-r * (T*0.9));

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
