to compute the code 
nvcc -o simulation simulation.cu -lcurand


  template <class O>
  struct UpAndOutCall : Option<O> {
    O s0, s1, barrier, k, r, sigma, T, dt, omega;
    float z, W1;
    int ind;
    O payoff, delta, vega, gamma, theta, rho;
    O lr_delta, lr_vega, lr_gamma, lr_theta, lr_rho;

    void PrintName() {
      printf("\n**OPTION** : UpAndOutCall\n");
    }

    __device__ void SimulatePaths(const int N, float *d_z) override {
    // Implement the path simulation logic here, specifically for the Up-and-Out Call
    // This is a simplified version assuming you're passing in the correct parameters
    z = d_z[ind]; // Assuming ind is correctly set to index into d_z
    
    bool knockedOut = false;
    float maxPrice = s0; // Starting price
    for (int i = 0; i < N; ++i) {
      float dW = sqrt(dt) * d_z[ind + i];
      s1 = s1 * exp((r - 0.5 * sigma * sigma) * dt + sigma * dW); // SDE for price
      if (s1 > barrier) {
        knockedOut = true;
        break;
      }
      if (s1 > maxPrice) maxPrice = s1;
    }

    payoff = knockedOut ? O(0) : exp(-r * T) * max(s1 - k, O(0));
    }

    __device__ void CalculatePayoffs(Greeks<double>& greeks) override {
      // Greeks are calculated under the assumption the option has not knocked out
      if (payoff > O(0.0)) {
        O d1 = (log(s0 / k) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        O d2 = d1 - sigma * sqrt(T);

        // Delta
        delta = exp(-r * T) * normcdf(d1);

        // Vega
        vega = s0 * exp(-r * T) * N_PDF(d1) * sqrt(T);

        // Gamma
        gamma = N_PDF(d1) / (s0 * sigma * sqrt(T));

        // Theta
        theta = -((s0 * N_PDF(d1) * sigma * exp(-r * T)) / (2 * sqrt(T))) - r * k * exp(-r * T) * normcdf(d2);


        // Likelihood Ratios
        lr_delta = payoff * (z / (s0 * sigma * sqrt(T)));
        lr_vega = payoff * (((z * z - 1) / sigma) - z / sqrt(T));
        lr_gamma = 2 * lr_vega / s0; // Simplified for demonstration
        lr_theta = -r * (lr_delta * s0 + payoff);
      } else {
        delta = vega = gamma = theta  = O(0.0);
        lr_delta = lr_vega = lr_gamma = lr_theta = O(0.0);
      }

      // Store results
      greeks.price[ind] = payoff;
      greeks.delta[ind] = delta;
      greeks.vega[ind] = vega;
      greeks.gamma[ind] = gamma;
      greeks.theta[ind] = theta;
      greeks.lr_delta[ind] = lr_delta;
      greeks.lr_vega[ind] = lr_vega;
      greeks.lr_gamma[ind] = lr_gamma;
      greeks.lr_theta[ind] = lr_theta;
    }

    __device__ void SimulatePathsQuasiBB(const int N, float *d_z, O *d_path) {
      float dt = T / N;
      float omega = r - 0.5 * sigma * sigma;
      bool knockedOut = false;

      // Initialize variables for the path
      float s_current = s0;
      float maxPrice = s0;

      // Generate the entire path upfront using Brownian Bridging
      float W = 0.0; // Accumulated Brownian motion
      for (int i = 0; i < N; ++i) {
          float dW = sqrt(dt) * d_z[ind + i]; // Incremental Brownian motion
          W += dW; // Accumulate Brownian motion
          float t = dt * (i + 1);
          float bridgeFactor = t * (T - t) / T; // Brownian Bridge adjustment factor

          // Apply the Brownian Bridge to adjust the path
          float adjustedW = W + bridgeFactor * (d_z[N] - W * T / t); // Final step uses d_z[N] for the end point
          float s_tilde = s0 * exp(omega * t + sigma * adjustedW); // Adjusted stock price

          // Check for barrier breach
          if (s_tilde > barrier) {
              knockedOut = true;
              break;
          }


    __host__ void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt, float sigma, O s0, O k, O barrier, float T, float omega, Greeks<double> &results) {
        ind = 0;

        for (int i = 0; i < NPATHS; ++i) {
            // Initial setup
            float z = h_z[ind]; 
            float z1 = z; // Capture z1 for lr_estimate
            float W1 = sqrt(dt) * z;
            float W_tilde = W1;
            float s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
            float s1 = s0 * exp(omega * dt + sigma * W1);
            bool knockedOut = false;

            // Check if the initial price is already above the barrier
            if (s1 > barrier) {
                knockedOut = true;
            }

            // Simulate over rest of N timesteps
            for (int n = 1; n < N && !knockedOut; n++) { 
                ind++;
                z = h_z[ind]; 

                // Stock dynamics
                W_tilde = W_tilde + sqrt(dt) * z;
                s_tilde = s0 * exp(omega * (dt * n - dt) + sigma * (W_tilde - W1));
                s1 = s_tilde * exp(omega * dt + sigma * W1); 

                // Check for barrier breach
                if (s1 > barrier) {
                    knockedOut = true;
                    break;
                }
            }

            // Calculate payoff based on whether the option was knocked out
            float payoff = 0.0;
            if (!knockedOut) {
                payoff = exp(-r * T) * max(s1 - k, O(0.0));
            }

            // Placeholder for Greeks calculation: Adjust as per your specific needs
            float delta = 0.0; // Calculate delta
            float vega = 0.0;  // Calculate vega
            float gamma = 0.0; // Calculate gamma
            float theta = 0.0; // Calculate theta
            

            // Store results
            results.price[i] = payoff;
            results.delta[i] = delta;
            results.vega[i] = vega;
            results.gamma[i] = gamma;
            results.theta[i] = theta;
            results.lr_delta[i] = 0.0;
            results.lr_vega[i] = 0.0;
            results.lr_gamma[i] = 0.0;
            results.lr_theta[i] = 0.0;
        }
      }
  };