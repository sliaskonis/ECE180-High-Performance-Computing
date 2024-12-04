# Variables
CUDAFLAGS = -I./src
CFLAGS = -O4 -g
LDFLAGS = -lm

# Source files
SRC = src/histogram-equalization_gpu.cu src/contrast-enhancement.c src/histogram-equalization.c src/main.c

# Output executable
OUTPUT = main

# Rules
all: $(OUTPUT)

$(OUTPUT): $(SRC)
	nvcc $(CUDAFLAGS) -Xcompiler="$(CFLAGS)" -Xlinker="$(LDFLAGS)" -o $@ $^

clean:
	rm main *.pgm

.PHONY: all clean