#pragma once

namespace qmc {

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

    __host__ void CopyFromDevice(const int size, const Greeks<T> &d_greeks) {
        cudaMemcpy(price, d_greeks.price,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(delta, d_greeks.delta,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(vega, d_greeks.vega,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(gamma, d_greeks.gamma,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(theta, d_greeks.theta,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_delta, d_greeks.lr_delta,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_vega, d_greeks.lr_vega,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_gamma, d_greeks.lr_gamma,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
        cudaMemcpy(lr_theta, d_greeks.lr_theta,
              sizeof(T) * size, cudaMemcpyDeviceToHost) ;
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
        double p_sum = 0.0, p_sum2 = 0.0, d_sum = 0.0, d_sum2 = 0.0,
               v_sum = 0.0, v_sum2 = 0.0, g_sum = 0.0, g_sum2 = 0.0, 
               t_sum = 0.0, t_sum2 = 0.0;
        double lr_d_sum = 0.0, lr_d_sum2 = 0.0, lr_v_sum = 0.0,
               lr_v_sum2 = 0.0, lr_g_sum = 0.0, lr_g_sum2 = 0.0,
               lr_t_sum = 0.0, lr_t_sum2 = 0.0;

        for (int i = 0; i < size; ++i) {
          p_sum += price[i];
          p_sum2 += price[i] * price[i];
          d_sum += delta[i];
          d_sum2 += delta[i] * delta[i];
          v_sum += vega[i];
          v_sum2 += vega[i] * vega[i];
          g_sum += gamma[i];
          g_sum2 += gamma[i] * gamma[i];
          t_sum += theta[i];
          t_sum2 += theta[i] * theta[i];
          lr_d_sum += lr_delta[i];
          lr_d_sum2 += lr_delta[i] * lr_delta[i];
          lr_v_sum += lr_vega[i];
          lr_v_sum2 += lr_vega[i] * lr_vega[i];
          lr_g_sum += lr_gamma[i];
          lr_g_sum2 += lr_gamma[i] * lr_gamma[i];
          lr_t_sum += lr_theta[i];
          lr_t_sum2 += lr_theta[i] * lr_theta[i];

        }

        avg_price = p_sum / size;
        avg_delta = d_sum / size;
        avg_vega = v_sum / size;
        avg_gamma = g_sum / size;
        avg_theta = t_sum / size;
        avg_lr_delta = lr_d_sum / size;
        avg_lr_vega = lr_v_sum / size;
        avg_lr_gamma = lr_g_sum / size;
        avg_lr_theta = lr_t_sum/size;

        err_price = sqrt((p_sum2 / size - (p_sum / size) * (p_sum / size)) / size);
        err_delta = sqrt((d_sum2 / size - (d_sum / size) * (d_sum / size)) / size);
        err_vega = sqrt((v_sum2 / size - (v_sum / size) * (v_sum / size)) / size);
        err_gamma = sqrt((g_sum2 / size - (g_sum / size) * (g_sum / size)) / size);
        err_theta = sqrt((t_sum2 / size - (t_sum / size) * (t_sum / size)) / size);
        err_lr_delta = sqrt((lr_d_sum2 / size - (lr_d_sum / size) * (lr_d_sum / size)) / size);
        err_lr_vega = sqrt((lr_v_sum2 / size - (lr_v_sum / size) * (lr_v_sum / size)) / size);
        err_lr_gamma = sqrt((lr_g_sum2 / size - (lr_g_sum / size) * (lr_g_sum / size)) / size);
        err_lr_theta = sqrt((lr_t_sum2 / size - (lr_t_sum / size) * (lr_t_sum / size)) / size);
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
}