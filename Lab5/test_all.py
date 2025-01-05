import subprocess
import sys

# Define the parameters for execution
implementations = ["serial"] #, "openmp", "cuda"]
iterations = sys.argv[1]
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
                    [sys.executable, script_path, impl, str(iterations), str(num_part)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error executing {script_path} with parameters {impl}, {num_part}")
                print(f"Error details: {e}")
            print("=" * 60)