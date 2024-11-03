import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_average(values):
    # Remove the highest and lowest value, then calculate the average of the remaining values
    if len(values) > 2:
        values = sorted(values)[1:-1]  # Remove the highest and lowest values
    return np.mean(values) if values else None  # Return average, or None if list is empty

def plot_file_averages(directory="results"):
    # Get all files in the specified directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Prepare to store the average value for each file
    averages = {}
    
    # Read each file, process values, and calculate the average
    for file in files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            values = [float(line.strip()) for line in f.readlines()]
            avg = calculate_average(values)
            if avg is not None:
                averages[file] = avg

    # Sort the averages from highest to lowest
    sorted_averages = dict(sorted(averages.items(), key=lambda item: item[1], reverse=True))

    # Plot the sorted averages as a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_averages.keys(), sorted_averages.values(), color='skyblue')
    
    # Add labels and title
    plt.xlabel("Files")
    plt.ylabel("Average Value (Excluding Highest and Lowest)")
    plt.title("Comparison of Averages from Files (Sorted)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Run the function to plot the averages
plot_file_averages()
