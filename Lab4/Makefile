# Variables
CUDAFLAGS = -I./src 
CFLAGS = -O4 
LDFLAGS = -lm

# Source files
SRC = src/histogram-equalization_gpu.cu src/main.cu src/contrast-enhancement.cu src/histogram-equalization.c

# Output executable
OUTPUT = main

# Rules
all: $(OUTPUT)

$(OUTPUT): $(SRC)
	nvcc $(CUDAFLAGS) -Xcompiler="$(CFLAGS)" -Xlinker="$(LDFLAGS)" -o $@ $^

clean:
	rm main *.pgm

.PHONY: all clean