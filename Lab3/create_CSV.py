import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def process_results(program_name, results_dir="results"):
    # Regex to match files with the naming convention <program_name>_<size>_out.txt
    file_pattern = re.compile(rf"^{program_name}_(\d+)_out\.txt$")
    
    # Initialize dictionaries to store times by size
    cpu_times = {}
    gpu_times = {}

    # Traverse the results directory and process files
    for filename in os.listdir(results_dir):
        match = file_pattern.match(filename)
        if match:
            size = int(match.group(1))
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as file:
                for line in file:
                    if line.startswith("CPU Execution Time:"):
                        cpu_time = float(line.split(":")[1].strip())
                        cpu_times.setdefault(size, []).append(cpu_time)
                    elif line.startswith("GPU Execution Time:"):
                        gpu_time = float(line.split(":")[1].strip())
                        gpu_times.setdefault(size, []).append(gpu_time)

    # Compute mean and standard deviation for each size
    sizes = sorted(cpu_times.keys())  # Ensure sizes are sorted
    cpu_means = [np.mean(cpu_times[size]) for size in sizes]
    cpu_stds = [np.std(cpu_times[size]) for size in sizes]
    gpu_means = [np.mean(gpu_times[size]) for size in sizes]
    gpu_stds = [np.std(gpu_times[size]) for size in sizes]

    df = pd.DataFrame(
        {   "sizes": sizes,
            "cpu_means": cpu_means,
            "cpu_stds": cpu_stds,
            "gpu_means": gpu_means,
            "gpu_stds": gpu_stds
        }
    )

    df.to_csv(f"results_csv/{program_name}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_CSV.py <program_name>")
        sys.exit(1)

    program_name = sys.argv[1]
    process_results(program_name)