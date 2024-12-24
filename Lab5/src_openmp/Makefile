CC = gcc
CFLAGS = -std=c99 -O3 -fopenmp -D_DEFAULT_SOURCE
LDFLAGS = -lm

TARGET = nbody

# Source files
SRC = nbody.c

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean up build artifacts
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean