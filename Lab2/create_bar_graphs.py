import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def read_times_from_file(filename):
    """Reads times from a file and returns them as a list of floats."""
    with open(filename, 'r') as file:
        times = [float(line.strip()) for line in file]
    return times

def parse_filename(filename):
    """Parses the filename to extract the parallel number and number of threads."""
    parts = filename.split('_')
    parallel_number = parts[1]
    num_threads = parts[3].replace('.txt', '')
    return parallel_number, int(num_threads)

def gather_data(directory):
    """Reads all files and organizes times by parallel number and thread count."""
    data = {}
    files = glob.glob(os.path.join(directory, 'parallel_*_times*.txt'))
    
    for file in files:
        parallel_number, num_threads = parse_filename(os.path.basename(file))
        times = read_times_from_file(file)
        
        if parallel_number not in data:
            data[parallel_number] = {}
        data[parallel_number][num_threads] = times

    return data

def clean_times(times):
    """Removes the highest and lowest time and returns the mean of the remaining times."""
    if len(times) > 2:
        # Remove the highest and lowest values
        times = sorted(times)[1:-1]
    return np.mean(times)

def plot_data(data):
    """Creates vertical bar plots for each parallel number, showing mean across different thread counts."""
    color = "#1984c5"  # Use the specified color for bars and labels
    lighter_black = "#4F4F4F"  # Lighter black color for labels
    light_grey = "lightgrey"  # Light grey color for x-axis line
    
    for parallel_number, times_by_threads in data.items():
        plt.figure(figsize=(10, 6))
        num_threads = sorted(times_by_threads.keys())
        
        # Calculate mean for each num_threads group
        means = []
        
        for threads in num_threads:
            mean = clean_times(times_by_threads[threads])
            means.append(mean)

        positions = range(len(num_threads))
        
        # Create vertical bar plot with the specified color
        bars = plt.bar(positions, means, color=color, alpha=0.9, zorder=2)
        
        plt.xticks(positions, [f"{threads}" for threads in num_threads])
        plt.ylabel("Execution Time (sec)", fontsize=12, color=lighter_black, labelpad=15)  # Set labelpad for y label
        plt.xlabel("Number of Threads", fontsize=12, color=lighter_black, labelpad=15)  # Set labelpad for x label
        plt.title(f"Parallel K-Means Execution Times (parallel_kmeans_{parallel_number}.c)", fontsize=12)  # Set lighter black color for title
        
        # Remove the lines of the axes but keep the labels
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)  # Hide left spine
        
        # Restore the bottom spine (x-axis line) and set its color to light grey
        plt.gca().spines['bottom'].set_color(light_grey)  # Set x-axis color to light grey
        plt.gca().spines['bottom'].set_linewidth(1)  # Optional: adjust the width of the x-axis line
        
        # Add horizontal grid lines in light grey
        plt.grid(axis='y', color='lightgrey', linestyle='--', linewidth=0.7, zorder=1)

        # Add bold data labels on top of the bars with the same color
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", 
                     ha='center', va='bottom', fontweight='bold', color=color)

        # Remove tick marks from x and y axes
        plt.tick_params(axis='x', which='both', length=0)  # Remove x ticks
        plt.tick_params(axis='y', which='both', length=0)  # Remove y ticks

        plt.yticks(color='black')  # Keep y-axis tick labels
        plt.xticks(color='black')  # Keep x-axis tick labels
        
        plt.show()

if __name__ == "__main__":
    directory = './results'  # Set to the directory where your files are stored
    data = gather_data(directory)
    plot_data(data)
