/******************************************************************************
 *                                                                            *
 *  File:       conv2d_2D_2D.cu                                              *
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
 __global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
                        int imageW, int imageH, int filterR) {
     int tx = (blockIdx.x * blockDim.x) + threadIdx.x;
     int ty = (blockIdx.y * blockDim.y) + threadIdx.y;
     double sum=0;
 
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
  *          GPU Column Convolution        *
  ******************************************/
 __global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter,
                    int imageW, int imageH, int filterR) {
    int tx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ty = (blockIdx.y * blockDim.y) + threadIdx.y;
    double sum=0;
 
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
 
 /***************************************
  *           CPU Row Convolution       *
  ***************************************/
 __host__ void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                        int imageW, int imageH, int filterR) {
 
    int x, y, k;
                       
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            double sum = 0;
 
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
 
 /***************************************
  *           CPU Row Convolution       *
  ***************************************/
 __host__ void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
                    int imageW, int imageH, int filterR) {
 
    int x, y, k;
   
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            double sum = 0;
 
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
 
 /*
 ------------------------------ END OF FUNCTION DECLARATION ------------------------------
 */
 
 int main(int argc, char **argv) {
     
    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    double
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

    /**********************************************************/
    /*                   Host Memory Allocation               */
    /**********************************************************/
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    assert(h_Filter != NULL);
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    assert(h_Input != NULL);
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    assert(h_Buffer != NULL);
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    assert(h_OutputCPU != NULL);
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));
    assert(h_OutputGPU != NULL);

    /**********************************************************/
    /*                   Memory Initialization                */
    /**********************************************************/
    srand(200);
    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    }
    
    /**********************************************************/
    /*                   Host Code Execution                  */
    /**********************************************************/

    clock_t start = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);          // Row convolution       
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);   // Column convolution
    clock_t end = clock();

    double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;

    /**********************************************************/
    /*                   Device Memory Allocation             */
    /**********************************************************/
    cudaMalloc((void**) &d_Filter, FILTER_LENGTH*sizeof(double));
    cudaMalloc((void**) &d_Input, imageW*imageH*sizeof(double));
    cudaMalloc((void**) &d_Buffer, imageW*imageH*sizeof(double));
    cudaMalloc((void**) &d_OutputGPU, imageW*imageH*sizeof(double));

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
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH*sizeof(double),cudaMemcpyHostToDevice);                           
    cudaMemcpy(d_Input, h_Input, imageW*imageH*sizeof(double),cudaMemcpyHostToDevice);                             

    // Perform row-wise convolution on the GPU
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);  
    err = checkCudaError("convolutionRowGPU");

    // Synchronize the device to ensure the row convolution is complete
    cudaDeviceSynchronize();

    // Perform column-wise convolution on the GPU
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);  
    err = checkCudaError("convolutionColumnGPU");

    // Copy the resulting data back to the host
    cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW*imageH*sizeof(double),cudaMemcpyDeviceToHost);                    
    
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
        for (i = 0; i < imageW * imageH; i++) {
            double error = ABS(h_OutputCPU[i] - h_OutputGPU[i]);
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