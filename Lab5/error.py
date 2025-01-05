import math
import sys
from decimal import Decimal

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
    if diff == 0:
        return "No difference"  # Numbers are exactly the same
    # Count the number of leading zeros after the decimal point
    err = -math.floor(math.log10(diff))-1
    if not max_error:
        max_error = err
    if err<max_error:
        max_error = err
    return err

# Open the file and read its lines
with open(filename, 'r') as file:
    for line in file:
        # Split the line into two numbers and convert them to floats
        num1, num2 = line.strip().split()
        num1 = Decimal(num1)
        num2 = Decimal(num2)
        #print(num1,", ", num2)
        # Compare the numbers and increment errors if they are the same
        if num1 == num2:
            error_degrees.append("same")  # For numbers that are identical
            continue
        else:
            errors += 1

        # Find the degree of the error and store it in error_degrees
        if num1 != num2:
            error_degrees.append(find_error_degree(num1, num2))
        # else:
        #     error_degrees.append("same")  # For numbers that are identical

# Print the results
print(f"Number of errors : {errors}")
print(f"Max error at digit : {max_error}")
# print("Error degrees array: ", error_degrees)
