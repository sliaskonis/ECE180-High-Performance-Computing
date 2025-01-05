NVCC = nvcc
NVCCFLAGS = -arch=sm_30 -O3

TARGET = nbody

# Source files
SRC = nbody.cu

ifeq ($(SAVE_FINAL_COORDINATES),1)
    NVCCFLAGS += -DSAVE_FINAL_COORDINATES
endif

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