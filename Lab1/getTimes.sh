#!/bin/bash

# Check if the user provided a number of times to execute the program
if [ $# -ne 1 ]; then
  echo "Usage: $0 <number_of_executions>"
  exit 1
fi

# Assign the user input to N
N=$1

# Create or overwrite the CSV file
csv_file="times.csv"
# Clear the file initially
> "$csv_file"

# Loop through each subdirectory
for dir in */; do
  # Check if the directory contains a .c file
  c_file=$(find "$dir" -maxdepth 1 -name "*.c")
  
  if [ -n "$c_file" ]; then
    # Extract the base name of the C file (without the .c extension)
    base_name=$(basename "$c_file" .c)
    
    # Compile the C file
    icx -O0 "$c_file" -o "$dir/$base_name"
    
    if [ $? -eq 0 ]; then
      # Write the file name to the CSV file
      echo "$base_name" >> "$csv_file"
      
      # Run the compiled program N times and append the results to the CSV
      for (( i=1; i<=N; i++ )); do
        # Capture the execution time output (assuming it's printed to stdout)
        exec_time=$("$dir/$base_name")
        
        # Append the execution time to the CSV file
        echo "$exec_time" >> "$csv_file"
      done
      
      # Add a newline after each file's executions for readability
      echo "" >> "$csv_file"
    else
      echo "Compilation failed for $c_file"
    fi
  else
    echo "No C file found in $dir"
  fi
done

# Clean up the compiled programs
find . -type f -executable -delete
