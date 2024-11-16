/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy    0.00005 

/***************************************
 *   Reference Row Convolution Filter  *
 ***************************************/
__global__ void convolutionRowGPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
        float sum=0;
    for (int k = -filterR; k <= filterR; k++) {
        int d = tx + k;
        if (d >= 0 && d < imageW) {
          sum += h_Src[ty * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[ty * imageW + tx] = sum;
    }
}

/******************************************
 *   Reference Column Convolution Filter  *
 ******************************************/
__global__ void convolutionColumnGPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
        float sum=0;
    for (int k = -filterR; k <= filterR; k++) {
        int d = ty + k;
        if (d >= 0 && d < imageW) {
          sum += h_Src[d * imageW + tx] * h_Filter[filterR - k];
        }     

        h_Dst[ty * imageW + tx] = sum;
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
    cudaMalloc((void**) &d_Filter, imageW*sizeof(float));
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
    cudaMemcpy(d_Filter, h_Filter, imageH * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, imageW*imageH*sizeof(float),cudaMemcpyHostToDevice);
    
    /********************************** CPU Execution **********************************/
    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);          // Row convolution       
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);   // Column convolution

    /********************************** GPU Execution **********************************/
    printf("GPU computation...\n");

    cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW*imageH*sizeof(float),cudaMemcpyDeviceToHost);

    /**********************  Kernel Launch Configuration ***************************/
    dim3 dimGrid(1, 1);
    dim3 dimBlock(imageW, imageH);

    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    
    /********************** Verify Correctness **********************/
    printf("Verifying results...\n");
    float error = 0;
    for (i = 0; i < imageW * imageH; i++) {
        error = ABS(h_OutputCPU[i] - h_OutputGPU[i]);
        if (error > accuracy) {
            printf("Test failed\n");

            // Print arrays
            printf("h_OutputCPU\n");
            for (i =0; i < imageW; i++) {
              for (int j = 0; j < imageH; j++) {
                printf("%f ", h_OutputCPU[i * imageW + j]);
              }
              printf("\n");
            }
            printf("\n");

            printf("h_OutputGPU\n");
            for (i =0; i < imageW; i++) {
              for (int j = 0; j < imageH; j++) {
                printf("%f ", h_OutputGPU[i * imageW + j]);
              }
              printf("\n");
            }
            printf("\n");
            break;
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