from collections import Counter
import matplotlib.pyplot as plt
from decimal import Decimal
import math
import sys
import re

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

    # Calculate max error
    if max_error is None or diff > max_error:
        max_error = diff

    # Iterate through all the digits of the numbers and check if they are the same
    digit = 0
    diff_str = str(diff)

    integer_part, decimal_part = diff_str.split(".") if "." in diff_str else (diff_str, "0")
    # Find the first non-zero digit
    integer_digit = -1
    for i in range(len(integer_part)):
        if integer_part[i] != '0':
            integer_digit = i
            digit_val = int(integer_part[i])
            if digit_val > 5:
                integer_digit = +1
            return True, integer_digit

    if integer_digit == -1:
        decimal_digit = -1
        for i in range(len(decimal_part)):
            if decimal_part[i] != '0':
                decimal_digit = i
                digit_val = int(decimal_part[i])
                if digit_val >= 5:
                    decimal_digit = +1
                return False, decimal_digit



# Find the correct golden output file for comparison based on the number of particles
match = re.search(r'\d+', filename)
number = match.group()

# Define the golden file name
golden_filename = sys.argv[2]

error_degree_integer = []
error_degree_decimal = []
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
                integer_p, val = find_error_degree(num1, num2)
                if integer_p:
                    error_degree_integer.append(val)
                else:
                    error_degree_decimal.append(val)

                # Tollerance < .2
                if val < 2:
                    errors += 1

# Print the results
print(f"Number of errors : {errors}")

# Count how many erros occur for each digit
counts_int = Counter(error_degree_integer)
counts_dec = Counter(error_degree_decimal)

# Sort count
sorted_by_count_int = sorted(counts_int.items(), key=lambda x: x[0])
sorted_by_count_dec = sorted(counts_dec.items(), key=lambda x: x[0])

adjusted_by_count_int = [(key - len(sorted_by_count_int), count) for key, count in sorted_by_count_int]
sorted_by_count_int = [(f"{-int(key)}.", value) for key, value in adjusted_by_count_int]
sorted_by_count_dec = [(f".{int(key) + 1}", value) for key, value in sorted_by_count_dec]

print("Max error: ", max_error)
print("Frequency of errors for each digit:")
print(sorted_by_count_int)
print(sorted_by_count_dec)

sorted_by_count = sorted_by_count_int + sorted_by_count_dec

# Plot the frequency of errors for each digit
nums = []
counts_values = []
for num, count in sorted_by_count:
    nums.append(num)
    counts_values.append(count)
    print(f"{num}: {count}")

# Add the count values on top of the bars
for i, count in enumerate(counts_values):
    plt.text(
        nums[i],           # X-coordinate
        count + 0.5,       # Y-coordinate (just above the bar)
        f"{count}",        # Text to display
        ha='center',       # Horizontal alignment
        va='bottom',       # Vertical alignment
        fontsize=10,       # Font size
        fontweight='bold'  # Bold text
    )

# Plot the counts
plt.bar(nums, counts_values, color='grey', edgecolor='black')
plt.xlabel("Error Position (Digit in Integer/Decimal Parts)", fontweight = 'bold')
plt.ylabel("Frequency", fontweight = 'bold')
plt.title("Distribution of Error Frequencies by Digit Position", fontweight = 'bold')
plt.xticks(nums, fontweight = 'bold')  # Ensure all numbers appear on the x-axis
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder = 0)

# Show the plot
plt.show()