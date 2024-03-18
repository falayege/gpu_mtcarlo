**Pricing of different options**
The pricing is done in C++ (+cuda). The goal is to showcase the efficience of the GPU for the QMC method. Therefore the options are both priced on the CPU and on the GPU and the results are compared (in terms of rapidity and precision - thanks to the Likelihood ratio method). The Quasi Monte Carlo is done with Sobol Sequences for a better coverage of the space. The options priced are calls and the following one : Asian option, binary Asian, barrier.
to compute the code 
nvcc -o simulation simulation.cu -lcurand
