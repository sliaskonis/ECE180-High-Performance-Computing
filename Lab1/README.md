# Lab 1 - Sobel Filter Code Optimization

This directory contains all the code related to Lab 1, focusing on optimizing the Sobel filter implementation. The process follows a step-by-step optimization of the original code, with each stage stored in separate directories.

## Directory Structure

- `original_code/`: Contains the unoptimized, original version of the Sobel filter code.
- `01.optimization/` to `14.final_optimizations/`: Each directory contains an optimized version of the code. The number prefix on each folder indicates the order in which optimizations were applied. The final directory (`14.final_optimizations/`) includes all cumulative optimizations applied throughout the process.

Each optimization builds upon the previous version, so the latest version incorporates all prior improvements.

## Running Performance Tests

To measure and compare the execution times of each optimization, a script named `getTimes.sh` is provided. This script will run the code a specified number of times and record the execution times for analysis.

### 1. **Running the Performance Script**  
   To execute the performance test for any optimization, use the following command:

   ```bash
   bash ./getTimes.sh <number_of_runs> <optimization_level>
   ```
	- <number_of_runs>: The number of times you want to run each optimization for benchmarking.
	- <optimization_level>: Can be either -O0 (no optimizations) or -fast (maximum optimization level).


After collecting the execution times, you can visualize the results by running the getStats.py script, which generates a comparison graph.

1. **Execute script**:
    ```bash
    python getStats.py
    ```
