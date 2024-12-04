# Variables
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -I./src
CUDA_FLAGS = -I./src

# Source files
SRC_DIR = src
C_SRCS = $(SRC_DIR)/contrast-enhancement.c $(SRC_DIR)/histogram-equalization.c $(SRC_DIR)/main.c
CUDA_SRCS = $(SRC_DIR)/histogram-equalization_gpu.cu
HEADERS = $(SRC_DIR)/hist-equ.h

# Object files
OBJS = $(C_SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Output executable
OUTPUT = main

# Rules
all: $(OUTPUT)

$(OUTPUT): $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(OUTPUT)

.PHONY: all clean