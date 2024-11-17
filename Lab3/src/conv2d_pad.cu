/******************************************************************************
 *                                                                            *
 *  File:       conv2d_pad.cu                                                *
 *  Description:                                                              *
 *      This file contains the implementation of a 2D convolution using       *  
 *      a separable filter. The convolution is performed on the CPU as well   *
 *      as the GPU using CUDA.                                                *
 *      For the GPU implementation, the geometry used is a 2D grid of 2D      *
 *      thread blocks.                                                        *
 *                                                                            *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)

/****************** ACCURACY MACROS ******************/
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy    0.5

/****************** GRID/BLOCK GEOMETRY ******************/
#define TILE_WIDTH 32
#define GRID_X ceil((float)imageW/TILE_WIDTH)
#define GRID_Y ceil((float)imageH/TILE_WIDTH)

bool checkCudaError(const char *step) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in %s: %s\n", step, cudaGetErrorString(err));
        return true;
    }
    return false;
}

/*
------------------------------ CPU/GPU CONVOLUTION FUNCTION DECLARATION ------------------------------
*/

/***************************************
 *           GPU Row Convolution       *
 ***************************************/
__global__ void convolutionRowGPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {
    int tx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ty = (blockIdx.y * blockDim.y) + threadIdx.y;
    float sum=0;

    int xnew = tx + filterR;
    int ynew = ty + filterR;

    for (int k = -filterR; k <= filterR; k++) {
        int d = xnew + k;
        sum += h_Src[ynew * imageW + d] * h_Filter[filterR - k];
    }

    h_Dst[ynew * imageW + xnew] = sum;
}

/***************************************
 *           GPU Column Convolution    *
 ***************************************/
__global__ void convolutionColumnGPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {
    int tx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ty = (blockIdx.y * blockDim.y) + threadIdx.y;
    float sum=0;

    int xnew = tx + filterR;
    int ynew = ty + filterR;

    for (int k = -filterR; k <= filterR; k++) {
        int d = ynew + k;
        sum += h_Src[d * imageW + xnew] * h_Filter[filterR - k];
    }

    h_Dst[ynew * imageW + xnew] = sum;
}

/***************************************
 *           CPU Row Convolution       *
 ***************************************/
__host__ void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = filterR; y < imageH - filterR; y++) {
    for (x = filterR; x < imageW - filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        sum += h_Src[y * imageW + d] * h_Filter[filterR - k];   
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

/***************************************
 *           CPU Row Convolution       *
 ***************************************/
__host__ void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = filterR; y < imageH - filterR; y++) {
    for (x = filterR; x < imageW - filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;
        sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

/*
------------------------------ END OF FUNCTION DECLARATION ------------------------------
*/

int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    float
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

    int imageW;
    int imageH;
    unsigned int i;

    bool err;

    // We assume that imageW = imageH = N, where N is given by the user.
    if (argc != 3) {
        printf("Usage: %s <image size> <filter radius>\n", argv[0]);
        printf("Image size must be a power of 2\n");
        exit(1);
    }

    imageW = atoi(argv[1]);
    filter_radius = atoi(argv[2]);
    imageH = imageW;

    if (imageW < FILTER_LENGTH) {
        printf("Error: Filter length exceeds image dimensions\n");
        exit(1);
    }

    int padded_size = (imageW + 2*filter_radius) * (imageH + 2*filter_radius);
    int newImageW = imageW + (2 * filter_radius);
    int newImageH = imageH + (2 * filter_radius);

    /**********************************************************/
    /*                   Host Memory Allocation               */
    /**********************************************************/
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    assert(h_Filter != NULL);
    h_Input     = (float *)calloc(padded_size , sizeof(float));
    assert(h_Input != NULL);
    h_Buffer    = (float *)calloc(padded_size , sizeof(float));
    assert(h_Buffer != NULL);
    h_OutputCPU = (float *)calloc(padded_size , sizeof(float));
    assert(h_OutputCPU != NULL);
    h_OutputGPU = (float *)calloc(padded_size , sizeof(float));
    assert(h_OutputGPU != NULL);

    /**********************************************************/
    /*                   Memory Initialization                */
    /**********************************************************/
    srand(200);
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (int i = 0; i < imageH; i++) {
        for (int j = 0; j < imageW; j++) {
            h_Input[(i + filter_radius) * (imageW + 2 * filter_radius) + (j + filter_radius)] = 
                (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
        }
    }

    /**********************************************************/
    /*                   Host Code Execution                  */
    /**********************************************************/

    clock_t start = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, newImageW, newImageH, filter_radius);          // Row convolution       
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, newImageW, newImageH, filter_radius);   // Column convolution
    clock_t end = clock();

    double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;

    /**********************************************************/
    /*                   Device Memory Allocation             */
    /**********************************************************/
    cudaMalloc((void**) &d_Filter, FILTER_LENGTH*sizeof(float));
    cudaMalloc((void**) &d_Input, padded_size*sizeof(float));
    cudaMalloc((void**) &d_Buffer, padded_size*sizeof(float));
    cudaMalloc((void**) &d_OutputGPU, padded_size*sizeof(float));

    cudaMemset(d_Buffer, 0, padded_size*sizeof(float));
    cudaMemset(d_OutputGPU, 0, padded_size*sizeof(float));

    /**********************************************************/
    /*                   Device Code Execution                */
    /**********************************************************/

    dim3 dimGrid(GRID_X, GRID_Y);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    cudaEvent_t startGPU, stopGPU;
    float gpu_time;

    // Create CUDA events for timing purposes
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    // Start GPU timing
    cudaEventRecord(startGPU);
    
    // Copy filter and input data to device
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, padded_size*sizeof(float),cudaMemcpyHostToDevice);

    // Perform row-wise convolution on the GPU
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, newImageW, newImageH, filter_radius);  
    err = checkCudaError("convolutionRowGPU");
    
    // Synchronize the device to ensure the row convolution is complete
    cudaDeviceSynchronize();
    
    // Perform column-wise convolution on the GPU
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, newImageW, newImageH, filter_radius);
    err = checkCudaError("convolutionColumnGPU");

    // Copy the resulting data back to the host
    cudaMemcpy(h_OutputGPU, d_OutputGPU, padded_size*sizeof(float),cudaMemcpyDeviceToHost);

    // Stop GPU timing
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    // Calculate GPU execution time
    cudaEventElapsedTime(&gpu_time, startGPU, stopGPU);
    
    // Destroy CUDA events
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    /**********************************************************/
    /*                    Verify Correctness                  */
    /**********************************************************/
#ifdef VERIFY    
    if (!err) {
        
        printf("Verifying results...\n");

        int errors = 0;
        for (i = filter_radius; i < newImageH-filter_radius; i++) {
            for (int j = filter_radius; j < newImageW - filter_radius; j++) {
                float error = ABS(h_OutputCPU[i*newImageH+j] - h_OutputGPU[i*newImageH+j]);
                if (error > accuracy) {
                    errors++;
                    printf("Mismatch at index %d: CPU = %f, GPU = %f, Error = %f\n", 
                        i, h_OutputCPU[i], h_OutputGPU[i], error);
                }
            }
        }

        if (errors == 0) {
            printf("TEST PASSED\n");
        } else {
            printf("TEST FAILED with %d errors\n", errors);
        }
    }
#endif

#ifdef PRINT_TIMING
    printf("CPU Execution Time: %f\n", cpu_time);
    printf("GPU Execution Time: %f\n", gpu_time);
#endif

    // Free Host allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    
    // Free Device allocated memory
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_OutputGPU);

    // Reset the device and exit
    cudaDeviceReset();

    return 0;
}