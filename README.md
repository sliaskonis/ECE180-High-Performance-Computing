# High Performance Computing (HPC) Class Assignments

This repository contains the assignments and projects completed during my **High Performance Computing (HPC)** course at [University Name]. The coursework focuses on utilizing advanced computing techniques to solve complex, computationally intensive problems efficiently.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Assignments](#assignments)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The assignments cover a variety of topics central to HPC, including:
- **Parallel Programming**: Using multi-threading and distributed computing to maximize efficiency.
- **Optimization**: Analyzing and improving the performance of code.
- **Performance Analysis**: Profiling and benchmarking algorithms.
  
These assignments make use of popular tools and libraries, such as **MPI (Message Passing Interface)**, **OpenMP (Open Multi-Processing)**, and **CUDA (Compute Unified Device Architecture)**, to implement solutions in parallel environments.

## Technologies Used

The projects and assignments were built using the following tools and libraries:

- **C/C++** for core programming
- **MPI** for distributed computing
- **OpenMP** for shared memory multiprocessing
- **CUDA** for GPU-based parallel computing
- **GCC** compiler for C/C++ programs
- **Profiling tools** like `gprof`, `nvprof`, etc.

## Assignments

1. **Assignment 1: Introduction to Parallel Programming**
    - Implemented parallel algorithms using OpenMP.
    
2. **Assignment 2: Message Passing with MPI**
    - Explored communication between distributed systems using MPI.
    
3. **Assignment 3: GPU Programming with CUDA**
    - Accelerated performance for computational problems using GPU parallelism.

4. **Assignment 4: Performance Profiling and Optimization**
    - Optimized an application and profiled it using `gprof` and `nvprof`.

## Installation

To run these assignments on your local machine, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```

2. **Navigate to the assignment folder**:
    ```bash
    cd assignment-1
    ```

3. **Compile the code**:
    ```bash
    gcc -fopenmp program.c -o program
    ```

4. **Run the program**:
    ```bash
    ./program
    ```

For MPI or CUDA assignments, specific compilation instructions are provided within the respective assignment folders.

## Usage

Each assignment folder includes detailed instructions for compiling and running the code. Refer to the `README.md` inside each folder for assignment-specific details.


