#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include <cuda.h>
#include <curand.h>

#include "greeks.cuh"
#include "kernels.cuh"
#include "option.cuh"

using namespace qmc;
using namespace std;

class Timer {
    public:
      Timer() {
        cudaEventCreate(&d_start);
        cudaEventCreate(&d_stop);
      }

      inline void BeginDevice() {
        cudaEventRecord(d_start);
      }

      inline void StopDevice() {
        cudaEventRecord(d_stop);
        cudaEventSynchronize(d_stop);
        cudaEventElapsedTime(&d_elapsed, d_start, d_stop);
      } 

      float GetDeviceElapsedTime() {
        return d_elapsed;
      }

      inline void BeginHost() {
        h_start = chrono::steady_clock::now();
      }

      inline void StopHost() {
        h_stop = chrono::steady_clock::now();
        h_elapsed = chrono::duration_cast<chrono::milliseconds>(h_stop - h_start).count();
      }

      float GetHostElapsedTime() {
        return h_elapsed;
      }

      float GetSpeedUp() {
        return h_elapsed / d_elapsed;
      }

    private:
      cudaEvent_t d_start, d_stop;
      chrono::steady_clock::time_point h_start, h_stop;
      float h_elapsed, d_elapsed;

};


template <typename S>
void RunAndCompareMC(int npath, int timesteps, float h_T, float h_dt, float h_r,
    float h_sigma, float h_omega, float h_s0, float h_k, Method method,
    LikelihoodRatios<double>& lr_greeks) {

  if (method == Method::QUASI) {
    printf("Method : Quasi MonteCarlo\n");
  } else if (method == Method::STANDARD) {
    printf("Method : Standard\n");
  } else if(method == Method::QUASI_BB) {
    printf("Method : Quasi MonteCarlo with Brownian Bridging\n");
  }

  // Initalise host product and print name
  S h_option;
  h_option.PrintName();
  float *d_z, *d_temp_z;
  float *d_path;
  Greeks<double> h_greeks, d_greeks;
  Timer timer;

  // Copy values to GPU constants
  cudaMemcpyToSymbol(N, &timesteps, sizeof(timesteps));
  cudaMemcpyToSymbol(PATHS ,&npath, sizeof(npath));
  cudaMemcpyToSymbol(T, &h_T, sizeof(h_T));
  cudaMemcpyToSymbol(r, &h_r, sizeof(h_r));
  cudaMemcpyToSymbol(sigma, &h_sigma, sizeof(h_sigma));
  cudaMemcpyToSymbol(dt, &h_dt, sizeof(h_dt));
  cudaMemcpyToSymbol(omega, &h_omega, sizeof(h_omega));
  cudaMemcpyToSymbol(s0, &h_s0, sizeof(h_s0));
  cudaMemcpyToSymbol(k, &h_k, sizeof(h_k));

  // Allocate host and device results
  h_greeks.AllocateHost(npath);
  d_greeks.AllocateDevice(npath);

  // Allocate memory for random variables
  cudaMalloc((void **)&d_z, sizeof(float) * timesteps * npath);
  if (method == Method::QUASI || method == Method::QUASI_BB) {
    cudaMalloc((void **)&d_temp_z,
          sizeof(float) * timesteps * npath);
  }
  if (method == Method::QUASI_BB) {
    cudaMalloc((void **) &d_path,
          sizeof(float) * timesteps * npath);
  }

  timer.BeginDevice();

  curandGenerator_t gen;
  if (method == Method::STANDARD) {
     curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) ;
     curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) ;
    /*  curandGenerateNormal(gen, d_z, timesteps * npath, 0.0f, 1.0f) ); */
  } else if (method == Method::QUASI || method == Method::QUASI_BB) {
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(gen, timesteps);
    /* curandGenerateNormal(gen, d_temp_z, timesteps*npath, 0.0f, 1.0f)); */
  }

  timer.StopDevice();

  printf("CURAND timer (ms): %f\n\n",
          timer.GetDeviceElapsedTime());

  // Execute kernel and time it
  const int runs = 500;
  Greeks<double> final_greeks;
  final_greeks.AllocateHost(runs);

  // Perform M independent runs of the simulation to get variance
  for (int j = 0; j < runs; ++j) {
    // Transform ordering of random variables into one that maximises memory
    // locality when threads access for path simulation
    if (method == Method::QUASI || method == Method::QUASI_BB) {
      curandGenerateNormal(gen, d_temp_z, timesteps*npath, 0.0f, 1.0f);
      TransformSobol<<<npath/64, 64>>>(d_z, d_temp_z);
    } else {
       curandGenerateNormal(gen, d_z, timesteps * npath, 0.0f, 1.0f) ;
    }

    timer.BeginDevice();

    Simulation<S> <<<npath/64, 64>>>(d_z, d_path, d_greeks, method);

    timer.StopDevice();


    // Copy back results

    h_greeks.CopyFromDevice(npath, d_greeks);
    h_greeks.CalculateGreeks(npath);

    // Transfer averages to final results struct
    final_greeks.price[j] = h_greeks.avg_price;
    final_greeks.delta[j] = h_greeks.avg_delta;
    final_greeks.vega[j] = h_greeks.avg_vega;
    final_greeks.gamma[j] = h_greeks.avg_gamma;
    final_greeks.theta[j] = h_greeks.avg_theta;
    final_greeks.lr_delta[j] = h_greeks.avg_lr_delta;
    final_greeks.lr_vega[j] = h_greeks.avg_lr_vega;
    final_greeks.lr_gamma[j] = h_greeks.avg_lr_gamma;
    final_greeks.lr_theta[j] = h_greeks.avg_lr_theta;
    /* h_greeks.PrintGreeks(true ,"GPU"); */
  }

  final_greeks.CalculateGreeks(runs);

  // Grab the LR results to calculate later VRFs
  if (method == Method::STANDARD) {
    lr_greeks.delta = final_greeks.avg_lr_delta;
    lr_greeks.vega = final_greeks.avg_lr_vega;
    lr_greeks.gamma = final_greeks.avg_lr_gamma;
    lr_greeks.theta = final_greeks.avg_lr_theta;
    lr_greeks.err_delta = final_greeks.err_lr_delta;
    lr_greeks.err_vega = final_greeks.err_lr_vega;
    lr_greeks.err_theta = final_greeks.err_lr_theta;
    lr_greeks.err_gamma = final_greeks.err_lr_gamma;

    printf("\nLIKELIHOOD RATIO\n");
    printf("Delta: %10.5f (Error: %10.5f)\n", lr_greeks.delta, lr_greeks.err_delta);
    printf("Vega: %10.5f (Error: %10.5f)\n", lr_greeks.vega, lr_greeks.err_vega);
    printf("Gamma: %10.5f (Error: %10.5f)\n\n", lr_greeks.gamma, lr_greeks.err_gamma);
  }

  final_greeks.PrintGreeks(true, "GPU");

  printf("\nVRF for delta = %13.8f\n",
      (lr_greeks.err_delta * lr_greeks.err_delta)
      / (final_greeks.err_delta * final_greeks.err_delta));
  printf("\nVRF for vega = %13.8f\n",
      (lr_greeks.err_vega * lr_greeks.err_vega)
      / (final_greeks.err_vega * final_greeks.err_vega));
  printf("\nVRF for gamma = %13.8f\n",
      (lr_greeks.err_gamma * lr_greeks.err_gamma)
      / (final_greeks.err_gamma * final_greeks.err_gamma));

  // CPU calculation

  // Copy random variables
  float *h_z = (float *) malloc(sizeof(float) * timesteps * npath);
  float *h_temp_z = (float *) malloc(sizeof(float) * timesteps * npath);
  if (method == Method::QUASI || method == Method::QUASI_BB) {
     cudaMemcpy(h_temp_z, d_temp_z, sizeof(float) * timesteps * npath, cudaMemcpyDeviceToHost );

    // Rejig for sobol dimensions
    int i = 0, j = 0;
    while (i < npath) {
      while (j < timesteps) {
        h_z[i * timesteps + j] = h_temp_z[i + j * npath];
        j++;
      }
      i++;
      j = 0;
    }
  } else if (method == Method::STANDARD) {
     cudaMemcpy(h_z, d_z, sizeof(float) * timesteps * npath, cudaMemcpyDeviceToHost );
  }


  timer.BeginHost();
  h_option.HostMC(npath, timesteps, h_z, h_r, h_dt, h_sigma, h_s0, h_k, h_T,
      h_omega, h_greeks);
  timer.StopHost();

  h_greeks.CalculateGreeks(npath);
  /* h_greeks.PrintGreeks(false, "CPU"); */

  printf("\nGPU timer (ms): %f \n", timer.GetDeviceElapsedTime());
  printf("CPU timer (ms): %f \n", timer.GetHostElapsedTime());
  printf("Speedup factor: %fx\n", timer.GetSpeedUp());
  printf("\n------------------------------------------------------------------------\n");
  printf("------------------------------------------------------------------------\n");

   curandDestroyGenerator(gen) ;

  // Release memory and exit cleanly

  h_greeks.ReleaseHost();
  d_greeks.ReleaseDevice();
  final_greeks.ReleaseHost();
  free(h_z);
  free(h_temp_z);
   cudaFree(d_z) ;
  if (method == Method::QUASI || method == Method::QUASI_BB) {
     cudaFree(d_temp_z);
  }
  if (method == Method::QUASI_BB) {
     cudaFree(d_path);
  }
}

int main(int argc, const char **argv){
  int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices available" << std::endl;
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
  std::cout << "GPU Specifications\n";
  std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << std::endl;
  std::cout << "Maximum grid dimensions: (" << prop.maxGridSize[0] << ", " 
      << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
  std::cout << "Maximum number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Maximum thread dimensions: (" << prop.maxThreadsDim[0] << ", " 
      << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";


  std::cout << std::endl;
    
  int NPATHS = (1 << 15);
  int h_m = 6;

  while (h_m <= 8) {
    int h_N = 1 << h_m;
    float h_T, h_r, h_sigma, h_dt, h_omega, h_s0, h_k;

    LikelihoodRatios<double> lr_greeks;


    h_T     = 1.0f;
    h_r     = 0.1f;
    h_sigma = 0.2f;
    h_dt    = h_T/h_N;
    h_omega = h_r - (h_sigma * h_sigma) / 2.0f;
    h_s0      = 100.0f;
    h_k       = 90.0f;

    // Lookback option
    RunAndCompareMC<Lookback<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::STANDARD, lr_greeks);
    RunAndCompareMC<Lookback<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::QUASI, lr_greeks);
    RunAndCompareMC<Lookback<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::QUASI_BB, lr_greeks);
    printf("\n\n\n");

    // Arithmetic Asian option
    RunAndCompareMC<ArithmeticAsian<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::STANDARD, lr_greeks);
    RunAndCompareMC<ArithmeticAsian<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::QUASI, lr_greeks);
    RunAndCompareMC<ArithmeticAsian<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::QUASI_BB, lr_greeks);
    printf("\n\n\n");

    // Binary Asian option
    RunAndCompareMC<BinaryAsian<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::STANDARD, lr_greeks);
    RunAndCompareMC<BinaryAsian<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::QUASI, lr_greeks);
    RunAndCompareMC<BinaryAsian<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::QUASI_BB, lr_greeks);
    printf("\n\n\n");

    // Up And Out Barrier option
    RunAndCompareMC<UpAndOutCall<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, Method::STANDARD, lr_greeks);
    RunAndCompareMC<UpAndOutCall<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
      h_omega, h_s0, h_k, Method::QUASI, lr_greeks);
    RunAndCompareMC<UpAndOutCall<float>>(NPATHS, h_N, h_T, h_dt, h_r, h_sigma,
      h_omega, h_s0, h_k, Method::QUASI_BB, lr_greeks);
    printf("\n\n\n");



    /* NPATHS <<= 1; */
    h_m += 2; // Jump to 256 timesteps

  }

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

}

