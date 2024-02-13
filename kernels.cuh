#pragma once

#include <cuda.h>

#include "option.cuh"

namespace qmc {

  __host__ __device__ float N_PDF(float x) {
    return exp(-0.5f * x * x) * (1.0f / sqrt(2.0f * M_PI));
  }

  __host__ float N_CDF(float x) {
    return std::erfc(-x/std::sqrt(2))/2;
  }

  //Algorithm 2
  __global__ void TransformSobol(float *d_z, float *temp_z) {
    int desired_idx = threadIdx.x + N * blockIdx.x * blockDim.x;
    int temp_idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = 0; n < N; n++) {
      d_z[desired_idx] = temp_z[temp_idx];
      desired_idx += blockDim.x;
      temp_idx += PATHS;
    }
  }


  template <class S>
  __global__ void Simulation(float *d_z, float *d_path, Greeks<double> greeks,
      Method method) {
    S option;

    option.ind = threadIdx.x + N*blockIdx.x*blockDim.x;

    if (method == Method::QUASI_BB)
      option.SimulatePathsQuasiBB(N, d_z, d_path);
    else 
      option.SimulatePaths(N, d_z); 

    option.CalculatePayoffs(greeks);
  }

}
