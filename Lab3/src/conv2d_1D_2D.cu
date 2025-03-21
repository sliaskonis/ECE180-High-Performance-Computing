/*******************************************************************************
 *                                                                             *
 *  File:       conv2d_1D_2D.c                                                 *
 *  Description:                                                               *
 *      This file contains the implementation of a 2D convolution using        *  
 *      a separable filter. The convolution is performed on the CPU as well    *
 *      as the GPU using CUDA.                                                 *
 *      For the GPU implementation, the geometry used is one 2D thread block.  *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy    0.0000005 

/***************************************
 *   Reference Row Convolution Filter  *
 ***************************************/
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                       int imageW, int imageH, int filterR) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float sum=0;

    if (tx < imageW && ty < imageH) {
        for (int k = -filterR; k <= filterR; k++) {
            int d = tx + k;
            if (d >= 0 && d < imageW) {
                sum += d_Src[ty * imageW + d] * d_Filter[filterR - k];
            }     
        }
        d_Dst[ty * imageW + tx] = sum;  
    }
}

/******************************************
 *   Reference Column Convolution Filter  *
 ******************************************/
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
    			   int imageW, int imageH, int filterR) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float sum=0;

    if (tx < imageW && ty < imageH) {
        for (int k = -filterR; k <= filterR; k++) {
            int d = ty + k;
            if (d >= 0 && d < imageW) {
                sum += d_Src[d * imageW + tx] * d_Filter[filterR - k];
            }     
        }
        d_Dst[ty * imageW + tx] = sum;  
    }
}

// Reference row convolution filter
__host__ void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            float sum = 0;

            for (k = -filterR; k <= filterR; k++) {
                int d = x + k;

                if (d >= 0 && d < imageW) {
                    sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
                }     

                h_Dst[y * imageW + x] = sum;
            }
        }
    }
}

// Reference column convolution filter
__host__ void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            float sum = 0;

            for (k = -filterR; k <= filterR; k++) {
                int d = y + k; // height + radius  ie:  h + (-2,-1,0,1,2)

                if (d >= 0 && d < imageH) {
                sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
                }   
 
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

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");

    /************************ Host memory allocation ************************/
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    assert(h_Filter != NULL);
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    assert(h_Input != NULL);
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    assert(h_Buffer != NULL);
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    assert(h_OutputCPU != NULL);
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
    assert(h_OutputGPU != NULL);

    /************************ Device memory allocation ************************/
    cudaMalloc((void**) &d_Filter, FILTER_LENGTH*sizeof(float));
    cudaMalloc((void**) &d_Input, imageW*imageH*sizeof(float));
    cudaMalloc((void**) &d_Buffer, imageW*imageH*sizeof(float));
    cudaMalloc((void**) &d_OutputGPU, imageW*imageH*sizeof(float));

    // Initialize Filter and Image.
    // Both filter and image are stored in row-major order and are initialized
    // with random values.
    srand(200);
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    /**********************  Copy Memory to Device ***************************/
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, imageW*imageH*sizeof(float),cudaMemcpyHostToDevice);
    
    /********************************** CPU Execution **********************************/
    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);          // Row convolution       
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);   // Column convolution

    /********************************** GPU Execution **********************************/
    printf("GPU computation...\n");

    /**********************  Kernel Launch Configuration ***************************/
    dim3 dimGrid(1, 1);
    dim3 dimBlock(imageW, imageH);

    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in convolutionRowGPU: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in convolutionColumnGPU: %s\n", cudaGetErrorString(err));
    }
    else {
        // Copy results from device to host
        cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW*imageH*sizeof(float),cudaMemcpyDeviceToHost);

        /********************** Verify Correctness **********************/
        printf("Verifying results...\n");

        int errors = 0;
        for (i = 0; i < imageW * imageH; i++) {
            float error = ABS(h_OutputCPU[i] - h_OutputGPU[i]);
            if (error > accuracy) {
                errors++;
                printf("Mismatch at index %d: CPU = %f, GPU = %f, Error = %f\n", 
                        i, h_OutputCPU[i], h_OutputGPU[i], error);
            }
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