import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np

def read_times_from_file(filename):
    """Reads times from a file and returns them as a list of floats."""
    with open(filename, 'r') as file:
        times = [float(line.strip()) for line in file]
    return times

def parse_filename(filename):
    """Parses the filename to extract the number of threads."""
    parts = filename.split('_')
    num_threads = int(parts[-1].replace('.txt', ''))
    return num_threads

def gather_data(base_name):
    """Reads all files matching the base name pattern and organizes times by thread count."""
    data = {}
    files = glob.glob(f"{base_name}_*.txt")
    
    for file in files:
        num_threads = parse_filename(os.path.basename(file))
        times = read_times_from_file(file)
        
        if num_threads not in data:
            data[num_threads] = []
        data[num_threads].append(times)

    return data

def clean_times(times):
    """Removes the highest and lowest time and returns the mean of the remaining times."""
    if len(times) > 2:
        # Remove the highest and lowest values
        times = sorted(times)[1:-1]
    return np.mean(times)

def plot_double_bars(data, base_names):
    """Creates double bar plots comparing means across different thread counts for multiple experiments."""
    colors = ["#1984c5", "#22a7f0"]  # Colors for each experiment
    lighter_black = "#4F4F4F"  # Lighter black color for labels
    light_grey = "lightgrey"  # Light grey color for x-axis line
    
    # Sort thread counts for consistent ordering
    num_threads = sorted(data.keys())

    # Calculate mean times for each experiment per thread count
    means = {threads: [clean_times(exp_data) for exp_data in data[threads]] for threads in num_threads}

    # Position bars side by side for each thread count
    positions = np.arange(len(num_threads))
    bar_width = 0.35  # Width of each bar

    plt.figure(figsize=(10, 6))

    for idx, base_name in enumerate(base_names):
        # Extract means for this experiment
        experiment_means = [means[threads][idx] if idx < len(means[threads]) else 0 for threads in num_threads]
        
        # Plot bars side by side
        plt.bar(positions + idx * bar_width, experiment_means, bar_width, color=colors[idx % len(colors)], label=base_name)

    plt.xticks(positions + bar_width / 2, [f"{threads}" for threads in num_threads])
    plt.ylabel("Execution Time (sec)", fontsize=12, color=lighter_black, labelpad=15)
    plt.xlabel("Number of Threads", fontsize=12, color=lighter_black, labelpad=15)
    plt.title(f"Execution Times Comparison across Experiments", fontsize=14)
    
    # Remove the lines of the axes but keep the labels
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    # Add x-axis line in light grey
    plt.gca().spines['bottom'].set_color(light_grey)
    plt.gca().spines['bottom'].set_linewidth(1)
    
    # Add horizontal grid lines in light grey
    plt.grid(axis='y', color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)

    # Add bold data labels on top of each bar
    for idx, base_name in enumerate(base_names):
        experiment_means = [means[threads][idx] if idx < len(means[threads]) else 0 for threads in num_threads]
        for i, yval in enumerate(experiment_means):
            plt.text(positions[i] + idx * bar_width, yval, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', color=colors[idx % len(colors)], fontsize=8)

    # Remove tick marks from x and y axes
    plt.tick_params(axis='x', which='both', length=0)
    plt.tick_params(axis='y', which='both', length=0)

    plt.yticks(color='black')
    plt.xticks(color='black')

    # Add legend for each experiment
    plt.legend(title="Experiments")

    plt.show()

if __name__ == "__main__":
    # Collect base names from command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <base_name1> <base_name2> ...")
        sys.exit(1)

    base_names = sys.argv[1:]
    all_data = {}

    # Gather data for each experiment base name
    for base_name in base_names:
        data = gather_data(base_name)
        for threads, times_list in data.items():
            if threads not in all_data:
                all_data[threads] = []
            all_data[threads].append([clean_times(times) for times in times_list])

    # Plot double bars
    plot_double_bars(all_data, base_names)
