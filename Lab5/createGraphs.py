import numpy as np
import sys
import re

if (len(sys.argv) != 2):
    print("Usage: python3 createGraphs.py <file_name>")
    exit(1)

output_file = sys.argv[1]
# Lists to store parsed data
total_times = []
interactions = []

# Regular expressions to match total time and interactions
time_pattern = re.compile(r"Total time: (\d+\.\d+)")  # Matches Total time: <number>
interactions_pattern = re.compile(r"(\d+\.\d+) Billion Interactions")  # Matches <number> Billion Interactions

# Open and read the file
with open(output_file, "r") as file:
    lines = file.readlines()

# Initialize variables to store iteration data
for line in lines:
    # Search for total time
    time_match = time_pattern.search(line)
    if time_match:
        total_times.append(float(time_match.group(1)))  # Append the total time to the list

    # Search for interactions
    interactions_match = interactions_pattern.search(line)
    if interactions_match:
        interactions.append(float(interactions_match.group(1)))  # Append the interactions to the list

# Display the results
print("Total Times for Each Iteration:", total_times)
print("Interactions for Each Iteration:", interactions)

# Calculate and display the average and standard deviation of the total times
if total_times:
    avg_time = np.mean(total_times)  
    std_time = np.std(total_times) 
    print(f"Average Total Time: {avg_time:.3f} seconds")
    print(f"Standard Deviation of Total Time: {std_time:.3f} seconds")
else:
    print("No total times available for calculation.")