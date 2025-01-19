import subprocess
import sys

if len(sys.argv) != 4:
    print("Usage: python test_all.py <implementation> <iterations> <save_final_coordinates>")
    print("  <implementation> = serial | openmp | cuda | all")
    print("  <save_final_coordinates> = true | false")
    exit(1)

# Check which implementations will be executed
if sys.argv[1] == "all":
    implementations = ["serial", "openmp", "cuda"]
else:
    implementations = [sys.argv[1]]

# Define number of iterations each implementation will be executed
iterations = sys.argv[2]
num_particles = [30000, 65536, 131072]

# Path to the getTimes.py script
script_path = "getTimes.py"

# Iterate over all combinations of parameters
for impl in implementations:
        for num_part in num_particles:
            print(f"Running {script_path} with implementation={impl}, num_particles={num_part}")
            try:
                # Call the script with the specified arguments
                subprocess.run(
                    [sys.executable, script_path, impl, str(iterations), str(num_part), str(sys.argv[3])],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing {script_path} with parameters {impl}, {num_part}")
                print(f"Error details: {e}")
            print("=" * 60)