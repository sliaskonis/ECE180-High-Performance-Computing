import csv
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt

# Function to process the CSV file and calculate average and standard deviation
def process_csv(file_path):
    # Dictionary to store execution times for each program
    program_times = defaultdict(list)
    
    # Read the CSV file
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        current_program = None
        for row in reader:
            if len(row) == 0:
                # Skip empty rows
                continue
            try:
                # Try to convert the row to a float (exec time)
                exec_time = float(row[0])
                if current_program:
                    program_times[current_program].append(exec_time)
            except ValueError:
                # If conversion fails, it's the program name
                current_program = row[0]
    
    # Calculate and store statistics (average time for each program)
    program_stats = []
    
    for program, times in program_times.items():
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        program_stats.append((program, avg_time, std_dev))
        print(f"Program: {program}")
        print(f"  Average Time: {avg_time:.6f} seconds")
        print(f"  Standard Deviation: {std_dev:.6f} seconds")
        print()

    return program_stats

# Function to plot the average execution times
def plot_averages(program_stats):
# Extract program names and average times for plotting
    programs = [item[0] for item in program_stats]
    avg_times = [item[1] for item in program_stats]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the line with blue dots at each point
    plt.plot(programs, avg_times, color='skyblue', linestyle='-', marker='o', markersize=8, markerfacecolor='blue', markeredgewidth=2)
    
    # Add labels and title
    plt.xlabel('Optimizations', fontsize=12)
    plt.ylabel('Average Execution Time (seconds)', fontsize=12)
    plt.title('Average Execution Times of Programs', fontsize=14)
    
    # Rotate program names for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Path to the CSV file
    csv_file_path = "times.csv"
    
    # Process the CSV file and calculate statistics
    program_stats = process_csv(csv_file_path)
    
    # Plot the average execution times
    plot_averages(program_stats)
