#!/bin/bash

cd opt_src
make 

# Directory to store results
mkdir -p ../results_new

# Array of thread counts
thread_counts=(1 4 8 14 28 56)

# Loop over each thread count
for threads in "${thread_counts[@]}"
do
    # Set OMP_NUM_THREADS for this iteration
    export OMP_NUM_THREADS=$threads

    # Run each parallel version 22 times and save results with thread count in filename
        for i in {1..17}
        do 
            ./par_main_6 -o -b -n 2000 -i ../Image_data/texture17695.bin
        done > ../results_new/parallel_6_times_${threads}.txt
done

# Clean up
make clean
