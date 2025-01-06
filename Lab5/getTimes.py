import subprocess
import sys
import os

if len(sys.argv) != 5:
    print("Usage: python getTimes.py <implementation> <iterations> <num_particles> <save_final_coordinates>")
    print("  <implementation> = serial | openmp | cuda")
    print("  <save_final_coordinates> = true | false")
    exit(1)

# Define implementation types
implType = {
    "serial": "src/src_orig",
    "openmp": "src/src_openmp",
    "cuda": "src/src_cuda"
}

# Define command for compilation 
make_command = ["make"]
if len(sys.argv) > 2 and sys.argv[4] == "true":
    make_command.append("SAVE_FINAL_COORDINATES=1")

# Define the directory for the executable
src_dir = implType.get(sys.argv[1])
nbody_executable = os.path.join(src_dir, "nbody")

# Navigate to the directory and run `make`
if not os.path.isdir(src_dir):
    print(f"Error: Directory {src_dir} does not exist.")
    exit(1)

# Create a directory to store the timing results
results_dir = "results"
num_particles_dir = os.path.join(results_dir, sys.argv[3])  # Subdirectory for the number of particles
os.makedirs(num_particles_dir, exist_ok=True)

try:
    print(f"Compiling {nbody_executable}...")
    subprocess.run(make_command, cwd=src_dir, check=True)
    print("Compile successful.")
except subprocess.CalledProcessError as e:
    print(f"Error during 'make': {e}")
    exit(1)

# Check if the executable was created
if not os.path.isfile(nbody_executable):
    print(f"Error: {nbody_executable} does not exist after 'make'.")
    exit(1)

# Execute the program multiple times
num_executions = int(sys.argv[2])
outputs = []

for i in range(num_executions):
    print(f"Running iteration {i+1}...")
    try:
        result = subprocess.run([nbody_executable, sys.argv[3]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        outputs.append(result.stdout)
        print(f"Iteration {i+1} output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution {i+1}: {e}")
        print(f"Standard Error:\n{e.stderr}")
        break

# Save outputs to a file in the results/num_particles directory
output_file = os.path.join(num_particles_dir, f"{sys.argv[1]}_times.txt")
with open(output_file, "w") as f:
    for i, output in enumerate(outputs, start=1):
        f.write(f"Iteration {i} Output:\n{output}\n{'='*40}\n")

print(f"Outputs saved to {output_file}")

# Make clean
try:
    print("Cleaning up...")
    subprocess.run(["make", "clean"], cwd=src_dir, check=True)
    print("Clean successful.")
except subprocess.CalledProcessError as e:
    print(f"Error during 'make clean': {e}")
    exit(1)