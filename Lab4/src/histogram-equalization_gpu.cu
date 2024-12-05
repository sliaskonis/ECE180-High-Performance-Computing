extern "C" {
    #include <stdio.h>
    #include <cuda_runtime.h>
    #include <time.h>
    #include "hist-equ.h"

    #define MAX_THREADS_PER_BLOCK 1024
    #define GRID_DIM ceil(((float)img_size/MAX_THREADS_PER_BLOCK))

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
	// __global__ void histogram_calc(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
    //     __shared__ int private_hist[256];

    //     int i = threadIdx.x + blockIdx.x*blockDim.x,
    //         stride = blockDim.x * gridDim.x;

    //     if (threadIdx.x < nbr_bin) {
    //         private_hist[threadIdx.x] = 0;
    //     }
    //     __syncthreads();
		
    //     while (i < img_size) {
    //         atomicAdd(&(private_hist[img_in[i]]), 1);
    //         i += stride;
    //     }
    //     __syncthreads();
        
    //     if (threadIdx.x < nbr_bin) {
    //         atomicAdd(&(hist_out[threadIdx.x]), private_hist[threadIdx.x]);
    //     }
    // }

    // Histogram calculation: naive implementation
    __global__ void histogram_calc(int *d_hist_out, unsigned char *d_img_in, int nbr_bin) {
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
		
        atomicAdd(&(d_hist_out[d_img_in[tid]]), 1);
    }

    // Histogram equalization application: naive implementation
    __global__ void histogram_equ(unsigned char *d_img_out, unsigned char *d_img_in, int *d_lut) {
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
        
        if (d_lut[d_img_in[tid]] > 255) {
            d_img_out[tid] = 255;
        }
        else {
            d_img_out[tid] = (unsigned char)d_lut[d_img_in[tid]];
        }
    }

	// Kernel wrapper
    void histogram_gpu(unsigned char *img_out, unsigned char *img_in,
                                int img_size, int nbr_bin) {
        int padding = 0, padded_size = 0;
        int i, cdf, min, d;
        int *lut = (int *)malloc(sizeof(int)*nbr_bin);
        int *hist_out = (int *)malloc(sizeof(int)*nbr_bin);

		unsigned char *d_img_in, *d_img_out;
        int *d_hist_out;
        int *d_lut;

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

        histogram_calc<<<grid, block>>>(d_hist_out, d_img_in, nbr_bin);
		cudaDeviceSynchronize();

		checkCudaError("Histogram calculation");

		// Copy calculated histogram back to host 
        cudaMemcpy(hist_out, d_hist_out, sizeof(int)*nbr_bin, cudaMemcpyDeviceToHost);

        // Clean histogram counts added by the padding elements
		// Padding elements are set to 0
        hist_out[0] = hist_out[0] - padding;
		
        // Construct the LUT by calculating the CDF
        cdf = 0;
        min = 0;
        i = 0;
        while(min == 0){
            min = hist_out[i++];
        }
        d = img_size - min;
        for(i = 0; i < nbr_bin; i ++){
            cdf += hist_out[i];
            lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
            if(lut[i] < 0){
                lut[i] = 0;
            }
        }    
        
        // Device memory allocation for histogram equalization application kernel
        cudaMalloc((void**) &d_lut,     sizeof(int)*nbr_bin);
        cudaMalloc((void**) &d_img_out, sizeof(unsigned char)*padded_size);
        
        cudaMemcpy(d_lut, lut, sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);
        
        cudaMemset(d_img_out, 0, sizeof(unsigned char)*padded_size);

        // Launch kernel
        histogram_equ<<<grid, block>>>(d_img_out, d_img_in, d_lut);

        // Copy img back to host
        cudaMemcpy(img_out, d_img_out, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);

        // Free non-wanted memory 
        cudaFree(d_lut);
        cudaFree(d_img_in);
        cudaFree(d_img_out);
        cudaFree(d_hist_out);

        // Reset the device
        cudaDeviceReset();
    }
}