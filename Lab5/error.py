from collections import Counter
from decimal import Decimal
import re
import sys

# Define the file name
filename = sys.argv[1]

# Initialize variables
errors = 0
error_degree_integer = []
error_degree_decimal = []
max_error = Decimal("0")

# Function to calculate the degree of the error
def find_error_degree(num1, num2):
    diff = abs(num1 - num2)
    global max_error

    # Update max error
    if diff > max_error:
        print(f"New max error: {diff} from {num1} and {num2}")
        max_error = diff

    diff_str = f"{diff:.10f}".rstrip("0")  # Keep precision, strip trailing zeros
    integer_part, _, decimal_part = diff_str.partition(".")
    
    # Check integer part
    for i, char in enumerate(integer_part):
        if char != "0":
            return True, -(i-len(integer_part))

    # Check decimal part
    for i, char in enumerate(decimal_part):
        if char != "0":
            return False, i

    return False, len(decimal_part)  # Default case: least significant digit

# Define the golden file name
golden_filename = sys.argv[2]

# Open the files and read their lines
with open(filename, "r") as file1, open(golden_filename, "r") as file2:
    for line1, line2 in zip(file1, file2):
        try:
            num1 = Decimal(line1.strip())
            num2 = Decimal(line2.strip())
        except Exception:
            continue  # Skip invalid lines

        # Compare the numbers
        if num1 != num2:
            is_integer, degree = find_error_degree(num1, num2)
            if is_integer:
                error_degree_integer.append(degree)
            else:
                error_degree_decimal.append(degree)

            # Increment errors if the degree is less than 2
            if degree < 2:
                errors += 1

# Count the errors
counts_int = Counter(error_degree_integer)
counts_dec = Counter(error_degree_decimal)

# Sort counts
sorted_by_count_int = sorted(counts_int.items(), reverse=True)
sorted_by_count_dec = sorted(counts_dec.items())

# Prepare data for plotting
labels_int = [f"{key}." for key, _ in sorted_by_count_int]
labels_dec = [f".{key + 1}" for key, _ in sorted_by_count_dec]
labels = labels_int + labels_dec

values_int = [count for _, count in sorted_by_count_int]
values_dec = [count for _, count in sorted_by_count_dec]
values = values_int + values_dec

print(labels)
print(values)