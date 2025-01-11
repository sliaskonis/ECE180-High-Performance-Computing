import math
import sys
import re
from decimal import Decimal
from collections import Counter
import matplotlib.pyplot as plt

# Define the file name
filename = sys.argv[1]

# Initialize variables
errors = 0
differences = []
error_degrees = []
max_error = None

# Function to calculate the degree of the error
def find_error_degree(num1, num2):
    diff = abs(num1 - num2)
    global max_error

    # Count the number of leading zeros after the decimal point
    err = -math.floor(math.log10(diff))-1
    if not max_error:
        max_error = err
    if err<max_error:
        max_error = err
    return err

# Find the correct golden output file for comparison based on the number of particles
match = re.search(r'\d+', filename)
number = match.group()
golden_filename = f"final_coordinates/golden/golden_coordinates_{number}.txt"

error_degree = []
# Open the file and read its lines
with open(filename, 'r') as file1:
    with open(golden_filename, 'r') as file2:
        for line1,line2 in zip(file1,file2):
            # Split the line into two numbers and convert them to floats
            num1 = line1.strip()
            num2 = line2.strip()
            num1 = Decimal(num1)
            num2 = Decimal(num2)

            # Compare the numbers and increment errors if they are not the same
            if num1 != num2:
                error_degree.append(find_error_degree(num1, num2))
                # Tollerance < .2
                if error_degree[-1] < 2:
                    errors += 1

# Print the results
print(f"Number of errors : {errors}")
print(f"Max error at digit : {max_error}")

# Count how many erros occur for each digit
counts = Counter(error_degree)

# Sort count
sorted_by_count = sorted(counts.items(), key=lambda x: x[0])

for num, count in sorted_by_count:
    print(f"{num}: {count}")

# Plot the counts
plt.bar(nums, counts_values, color='skyblue', edgecolor='black')
plt.xlabel("Numbers")
plt.ylabel("Frequency")
plt.title("Frequency of Errors for Each Number")
plt.xticks(nums)  # Ensure all numbers appear on the x-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()