/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define TILE_WIDTH 32
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy    0.05

bool checkCudaError(const char *step) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in %s: %s\n", step, cudaGetErrorString(err));
        return true;
    }
    return false;
}

/***************************************
 *   Reference Row Convolution Filter  *
 ***************************************/
__global__ void convolutionRowGPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {
    int tx = (blockIdx.x * blockDim.x) + threadIdx.x + filterR;
    int ty = (blockIdx.y * blockDim.y) + threadIdx.y + filterR;
    float sum=0;

      for (int k = -filterR; k <= filterR; k++) {
        int d = tx + k;
        sum += h_Src[ty * imageW + d] * h_Filter[filterR - k];     
      }
      h_Dst[ty * imageW + tx] = sum;  
  }

/******************************************
 *   Reference Column Convolution Filter  *
 ******************************************/
__global__ void convolutionColumnGPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {
    int tx = (blockIdx.x * blockDim.x) + threadIdx.x + filterR;
    int ty = (blockIdx.y * blockDim.y) + threadIdx.y + filterR;
    float sum=0;

      for (int k = -filterR; k <= filterR; k++) {
        int d = ty + k;
        sum += h_Src[d * imageW + tx] * h_Filter[filterR - k];     
      }
      h_Dst[ty * imageW + tx] = sum;  
}

// Reference row convolution filter
__host__ void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = filterR; y < imageH + filterR; y++) {
    for (x = filterR; x < imageW + filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        sum += h_Src[y * imageW + d] * h_Filter[filterR - k];   
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

// Reference column convolution filter
__host__ void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = filterR; y < imageH + filterR; y++) {
    for (x = filterR; x < imageW + filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;
        sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

// Main program
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
    /*                   Device Memory Allocation             */
    /**********************************************************/
    cudaMalloc((void**) &d_Filter, FILTER_LENGTH*sizeof(float));
    cudaMalloc((void**) &d_Input, padded_size*sizeof(float));
    cudaMalloc((void**) &d_Buffer, padded_size*sizeof(float));
    cudaMalloc((void**) &d_OutputGPU, padded_size*sizeof(float));

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

    /**********************  Copy Memory to Device ***************************/
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, padded_size*sizeof(float),cudaMemcpyHostToDevice);
    
    /**********************************************************/
    /*                   Host Code Execution                  */
    /**********************************************************/
    printf("CPU computation...\n");
    printf("CPU Input\n");
    for (i = 0; i < padded_size; i++) {
        if (i % (imageW + 2*filter_radius) == 0) {
            printf("\n");
        }
        printf("%f ", h_Input[i]);
    }
    printf("\n");
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);          // Row convolution       
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);   // Column convolution

    /**********************************************************/
    /*                   Device Code Execution                */
    /**********************************************************/
    printf("GPU computation...\n");

    dim3 dimGrid(ceil((float)imageW/TILE_WIDTH), ceil((float)imageH/TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);   // Row convolution
    err = checkCudaError("convolutionRowGPU");
    
    if (!err) {
        cudaDeviceSynchronize();
        convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
        err = checkCudaError("convolutionColumnGPU");
    }
    

    /**********************************************************/
    /*                    Verify Correctness                  */
    /**********************************************************/
    if (!err) {
        cudaMemcpy(h_OutputGPU, d_OutputGPU, padded_size*sizeof(float),cudaMemcpyDeviceToHost);
        printf("Verifying results...\n");

        int errors = 0;
        for (i = 0; i < padded_size; i++) {
            float error = ABS(h_OutputCPU[i] - h_OutputGPU[i]);
            if (error > accuracy) {
                errors++;
                printf("Mismatch at index %d: CPU = %f, GPU = %f, Error = %f\n", 
                    i, h_OutputCPU[i], h_OutputGPU[i], error);
            }
        }

        // printf the arrays
        printf("CPU Output\n");
        for (i = filter_radius; i < imageH + filter_radius; i++) {
            for (int j = filter_radius; j < imageW + filter_radius; j++) {
                printf("%f ", h_OutputCPU[i*imageW+j]);
            }
            printf("\n");
        }

        printf("GPU Output\n");
        for (i = filter_radius; i < imageH + filter_radius; i++) {
            for (int j = filter_radius; j < imageW + filter_radius; j++) {
                printf("%f ", h_OutputGPU[i*imageW+j]);
            }
            printf("\n");
        }

        if (errors == 0) {
            printf("TEST PASSED\n");
        } else {
            printf("TEST FAILED with %d errors\n", errors);
        }
    }

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