#!/bin/bash

# Check if the user provided the number of executions and a compilation flag
if [ $# -ne 2 ]; then
  echo "Usage: $0 <number_of_executions> <-O0|-fast>"
  exit 1
fi

# Assign the user inputs to variables
N=$1
compilation_flag=$2

# Ensure the compilation flag is either -O0 or -fast
if [[ "$compilation_flag" != "-O0" && "$compilation_flag" != "-fast" ]]; then
  echo "Error: Compilation flag must be either -O0 or -fast"
  exit 1
fi

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
    
    # Compile the C file with the specified flag
    icx "$compilation_flag" "$c_file" -o "$dir/$base_name"
    
    if [ $? -eq 0 ]; then
      # Write the file name and compilation flag to the CSV file
      echo "$base_name ($compilation_flag)" >> "$csv_file"
      
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
