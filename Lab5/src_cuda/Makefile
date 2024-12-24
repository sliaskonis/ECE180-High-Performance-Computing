NVCC = nvcc
NVCCFLAGS = -arch=sm_30 -O3

TARGET = nbody

# Source files
SRC = nbody.cu

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

# Clean up build artifacts
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean