#!/bin/bash

# Check if a program name and number of iterations are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <program_name> <number_of_iterations>"
    exit 1
fi

nvcc -O4 -o $1 $1.cu -DPRINT_TIMING

mkdir -p results

program=$1
iterations=$2

# Define the array of values for the second argument
sizes=(64 128 256 512 1024 2048 4096 8192 16384)

# Execute the program for the specified number of iterations and array values
for size in "${sizes[@]}"; do
    output_file="results/${program}_${size}_out.txt"
    # Clear the output file for the current size
    > $output_file

    for i in $(seq 1 $iterations); do
        echo "Iteration $i with size $size:" >> $output_file
        ./$program $size 16 >> $output_file
        echo >> $output_file
    done
done

rm $program

echo "Output of $iterations iterations for all sizes stored in separate files under results/"