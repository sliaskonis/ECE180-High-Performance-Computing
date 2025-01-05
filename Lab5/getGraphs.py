import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import os

def parse_and_calculate(file_name):
    total_times = []
    interactions = []

    # Regular expressions to match total time and interactions
    time_pattern = re.compile(r"Total time: (\d+\.\d+)")  # Matches Total time: <number>
    interactions_pattern = re.compile(r"(\d+\.\d+) Billion Interactions")  # Matches <number> Billion Interactions

    try:
        with open(file_name, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None, None, None, None

    # Parse the file
    for line in lines:
        # Search for total time
        time_match = time_pattern.search(line)
        if time_match:
            total_times.append(float(time_match.group(1)))  # Append the total time to the list

        # Search for interactions
        interactions_match = interactions_pattern.search(line)
        if interactions_match:
            interactions.append(float(interactions_match.group(1)))  # Append the interactions to the list

    # Calculate and return the averages and standard deviations
    if total_times and interactions:
        avg_time = np.mean(total_times)
        std_time = np.std(total_times)
        avg_interactions = np.mean(interactions)
        std_interactions = np.std(interactions)
        return avg_time, std_time, avg_interactions, std_interactions
    else:
        print("No data available for calculation.")
        return None, None, None, None

results_dir = "results"

serial_times = []
openmp_time = []
cuda_times = []

serial_std = []
openmp_std = []
cuda_std = []

# Loop through each subdirectory in the results directory
for subdir in os.listdir(results_dir):
    subdir_path = os.path.join(results_dir, subdir)

    # Check if it's a directory
    if os.path.isdir(subdir_path):
        print(f"Processing directory: {subdir}")

        # Loop through the specific files in the subdirectory
        for file_name in ["serial_times.txt", "openmp_times.txt", "cuda_times.txt"]:
            file_path = os.path.join(subdir_path, file_name)

            # Check if the file exists
            if os.path.isfile(file_path):
                avg_time, std_time, avg_interactions, std_interactions = parse_and_calculate(file_path)
            else:
                print(f"  File not found: {file_name} in {subdir}")
                exit(1)

            if file_name == "serial_times.txt":
                serial_times.append(avg_time)
                serial_std.append(std_time)
            elif file_name == "openmp_times.txt":
                openmp_time.append(avg_time)
                openmp_std.append(std_time)
            elif file_name == "cuda_times.txt":
                cuda_times.append(avg_time)
                cuda_std.append(std_time)

# Plot the data
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(serial_times))
plt.bar(index, serial_times, bar_width, yerr=serial_std, label="Serial")
plt.bar(index + bar_width, openmp_time, bar_width, yerr=openmp_std, label="OpenMP")
plt.bar(index + 2*bar_width, cuda_times, bar_width, yerr=cuda_std, label="CUDA")
plt.xlabel("Number of Particles")
plt.ylabel("Average Time (sec)")
plt.title("Average Time vs Problem Size")
plt.xticks(index + bar_width, ["30000", "65536", "131072"])
plt.legend()
plt.tight_layout()
plt.show()