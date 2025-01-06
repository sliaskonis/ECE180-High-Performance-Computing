# Lab5 - N-Body Simulation Using CUDA
This repository contains the three different implementations for N-Body Simulation:
- Serial: A straightforward C program that performs the N-Body simulation on the CPU.
- OpenMP: A parallelized version of the N-Body simulation using OpenMP for multi-threaded execution on the CPU.
- CUDA: A highly parallelized implementation leveraging CUDA to execute the N-Body simulation on GPUs for improved performance.

## **Executing each implementation**
Each implementation is located in the src directory, organized by its respective name.
You can execute each implementation by using the provided makefiles in their directory:

```bash 
make <flags>
```
The following compilation flag is available:
- SAVE_FINAL_COORDINATES: saves the final coordinates of each body at the end of the simulation to a text file. 

Then start the simulation using:

```bash 
./nbody <number of particles>
```

## **Scripts** 
This repository also contains some useful scripts for testing/profiling each implementation.

### **Test All Script**
The `test_all.py` script is designed to test various implementations of a specific algorithm. The script supports serial, OpenMP, CUDA, or all implementations, and provides options for configuring the number of iterations and saving final coordinates. When executing the script, the specified implementation will be executed for `<iterations>` number of times and its output will be saved to a text file under the results directory. The saved file contains information such as the execution time of each iteration and the average number of interactions per second for all bodies. These output files can be used by the `getGraphs.py` script in order to create comparison graphs between different implementations.

#### **Usage**

Run the script from the command line with the following syntax:

```bash
python3 test_all.py <implementation> <iterations> <save_final_coordinates>
```

1. Implementation:
    - serial: Run the serial implementation
    - openmp: Run the OpenMP implementation
    - cuda: Run the CUDA implementation
    - all: Run all the implementations sequentially
2. Iterations: 
    - The number of iterations to execute each implementation
3. Save final coordinates:
    - true: Save the final coordinates to a file
    - false: Do not save the final coordinates

### **Get Graphs Script**