#!/bin/bash

cd opt_src
make 

# Directory to store results
mkdir -p ../results

# Array of thread counts
thread_counts=(1 4 8 14 28 56)

# Loop over each thread count
for threads in "${thread_counts[@]}"
do
    # Set OMP_NUM_THREADS for this iteration
    export OMP_NUM_THREADS=$threads
    
    # Run the sequential version 22 times and save results with thread count in filename
    for i in {1..22}
    do 
        ./seq_main -o -b -n 2000 -i ../Image_data/color17695.bin
    done > ../results/seq_times_${threads}.txt

    # Run each parallel version 22 times and save results with thread count in filename
    for j in {1..8}
    do
        for i in {1..22}
        do 
            ./par_main_$j -o -b -n 2000 -i ../Image_data/color17695.bin
        done > ../results/parallel_${j}_times_${threads}.txt
    done
done

# Clean up
make clean
