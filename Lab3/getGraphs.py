import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt


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

    # Plot mean execution times (without error bars)
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_means, label="CPU Time", marker='o', color='b')  # Blue for CPU
    plt.plot(sizes, gpu_means, label="GPU Time", marker='o', color='r')  # Red for GPU
    plt.xlabel("Size")
    plt.ylabel("Mean Time (ms)")
    plt.title(f"Mean Execution Times for {program_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{program_name}_mean_execution_times.png")
    plt.show()

    # Plot standard deviations (without error bars, and GPU in red)
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_stds, label="CPU Std Dev", marker='o', color='b')  # Blue for CPU
    plt.plot(sizes, gpu_stds, label="GPU Std Dev", marker='o', color='r')  # Red for GPU
    plt.xlabel("Size")
    plt.ylabel("Standard Deviation (ms)")
    plt.title(f"Standard Deviation of Execution Times for {program_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{program_name}_std_execution_times.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_results.py <program_name>")
        sys.exit(1)

    program_name = sys.argv[1]
    process_results(program_name)

################# LOGARITHMIC GRAPH ####################
# import os
# import re
# import sys
# import numpy as np
# import matplotlib.pyplot as plt

# def process_results(program_name, results_dir="results"):
#     # Regex to match files with the naming convention <program_name>_<size>_out.txt
#     file_pattern = re.compile(rf"^{program_name}_(\d+)_out\.txt$")
    
#     # Initialize dictionaries to store times by size
#     cpu_times = {}
#     gpu_times = {}

#     # Traverse the results directory and process files
#     for filename in os.listdir(results_dir):
#         match = file_pattern.match(filename)
#         if match:
#             size = int(match.group(1))
#             filepath = os.path.join(results_dir, filename)
#             with open(filepath, "r") as file:
#                 for line in file:
#                     if line.startswith("CPU Execution Time:"):
#                         cpu_time = float(line.split(":")[1].strip())
#                         cpu_times.setdefault(size, []).append(cpu_time)
#                     elif line.startswith("GPU Execution Time:"):
#                         gpu_time = float(line.split(":")[1].strip())
#                         gpu_times.setdefault(size, []).append(gpu_time)

#     # Compute mean and standard deviation for each size
#     sizes = sorted(cpu_times.keys())  # Ensure sizes are sorted
#     cpu_means = [np.mean(cpu_times[size]) for size in sizes]
#     cpu_stds = [np.std(cpu_times[size]) for size in sizes]
#     gpu_means = [np.mean(gpu_times[size]) for size in sizes]
#     gpu_stds = [np.std(gpu_times[size]) for size in sizes]

#     # Plot mean execution times with error bars (standard deviation)
#     plt.figure(figsize=(10, 6))
#     plt.errorbar(sizes, cpu_means, yerr=cpu_stds, label="CPU Time", fmt='-o', color='b', capsize=5)
#     plt.errorbar(sizes, gpu_means, yerr=gpu_stds, label="GPU Time", fmt='-o', color='r', capsize=5)

#     plt.xlabel("Size")
#     plt.ylabel("Execution Time (ms)")
#     plt.title(f"Execution Times with Standard Deviation for {program_name}")
#     plt.legend()
#     plt.grid(True)
#     plt.yscale('log')  # Set the y-axis to logarithmic scale
#     plt.tight_layout()
#     plt.savefig(f"{program_name}_execution_times_with_std_log.png")
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python process_results.py <program_name>")
#         sys.exit(1)

#     program_name = sys.argv[1]
#     process_results(program_name)