extern "C" {
    #include <stdio.h>
    #include <cuda_runtime.h>
    #include <time.h>
    #include "hist-equ.h"

    #define MAX_THREADS_PER_BLOCK 1024
    #define GRID_DIM ceil(((float)img_size/MAX_THREADS_PER_BLOCK)/100)

	/****************************** Helper Functions ******************************/
	bool checkCudaError(const char *step) {
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error in %s: %s\n", step, cudaGetErrorString(err));
			return true;
		}
		return false;
	}

	/****************************** Kernels ******************************/

    // Histogram calculation: stride implementation
	__global__ void histogram_calc(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
        __shared__ int private_hist[256];

        int i = threadIdx.x + blockIdx.x*blockDim.x,
            stride = blockDim.x * gridDim.x;

        if (threadIdx.x < nbr_bin) {
            private_hist[threadIdx.x] = 0;
        }
        __syncthreads();
		
        while (i < img_size) {
            atomicAdd(&(private_hist[img_in[i]]), 1);
            i += stride;
        }
        __syncthreads();
        
        if (threadIdx.x < nbr_bin) {
            atomicAdd(&(hist_out[threadIdx.x]), private_hist[threadIdx.x]);
        }
    }

    // // Histogram calculation: naive implementation
    // __global__ void histogram_calc(int *hist_out, unsigned char *img_in, int nbr_bin) {
    //     int tid = threadIdx.x + blockIdx.x*blockDim.x;
		
    //     atomicAdd(&(hist_out[img_in[tid]]), 1);
    // }

	// Kernel wrapper
    void histogram_gpu(int *hist_out, unsigned char *img_in,
                                int img_size, int nbr_bin) {
        int padding = 0, padded_size = 0;
		
		unsigned char *d_img_in;
        int *d_hist_out;
                            
        // Allocate device memory 
        padding = (img_size%MAX_THREADS_PER_BLOCK) ? (MAX_THREADS_PER_BLOCK - (img_size%MAX_THREADS_PER_BLOCK)) : 0;

		padded_size = img_size + padding;
		cudaMalloc((void**) &d_img_in,	 sizeof(unsigned char)*padded_size);
        cudaMalloc((void**) &d_hist_out, sizeof(int)*nbr_bin);

        cudaMemset (d_img_in,   0, sizeof(unsigned char)*padded_size);
        cudaMemset (d_hist_out, 0, sizeof(int)*nbr_bin);
        
		cudaMemcpy(d_img_in, img_in, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 block(MAX_THREADS_PER_BLOCK, 1, 1);
        dim3 grid(GRID_DIM, 1, 1);

        clock_t start, end;

        start = clock();
        histogram_calc<<<grid, block>>>(d_hist_out, d_img_in, img_size, nbr_bin);
		cudaDeviceSynchronize();
        end = clock();

        double time = (double) (end-start)/ CLOCKS_PER_SEC;

        printf("Hist kernel time: %f\n", time);
		checkCudaError("Histogram calculation");

		// Copy calculated histogram back to host 
        cudaMemcpy(hist_out, d_hist_out, sizeof(int)*nbr_bin, cudaMemcpyDeviceToHost);

        // Clean histogram counts added by the padding elements
		// Padding elements are set to 0
        hist_out[0] = hist_out[0] - padding;
		
        // Free non-wanted memory 
        cudaFree(d_img_in);
		cudaFree(d_hist_out);

        // Reset the device
        cudaDeviceReset();
    }
}