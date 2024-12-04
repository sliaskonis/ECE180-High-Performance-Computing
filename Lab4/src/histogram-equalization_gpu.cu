extern "C" {
    #include <stdio.h>
    #include <cuda_runtime.h>
    #include <time.h>
    #include "hist-equ.h"

    #define MAX_THREADS_PER_BLOCK 1024
    #define GRID_DIM ceil((float)img_size/MAX_THREADS_PER_BLOCK)

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
	// __global__ void histogram_gpu(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
    //     __shared__ int private_hist[256];

    //     int tid = threadIdx.x + blockIdx.x*blockDim.x;

    //     if (threadIdx.x < 256) {
    //         private_hist[threadIdx.x] = 0;
    //     }
    //     __syncthreads();
		
    //     atomicAdd(&(private_hist[img_in[tid]]), 1);
    //     __syncthreads();
        
    //     if (threadIdx.x < 256) {
    //         atomicAdd(&(hist_out[threadIdx.x]), private_hist[threadIdx.x]);
    //     }
    // }
	// __global__ void histogram_gpu(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
    //     __shared__ unsigned int private_hist[256];

    //     int i = threadIdx.x + blockIdx.x*blockDim.x,
    //         stride = blockDim.x*gridDim.x;

    //     if (threadIdx.x < 256) {
    //         private_hist[threadIdx.x] = 0;
    //     }
    //     __syncthreads();

    //     while (i < img_size) {
    //         atomicAdd( &(private_hist[img_in[i]]), 1);
    //         i += stride;
    //     }
    //     __syncthreads();
        
    //     if (threadIdx.x < 256) {
    //         atomicAdd(&(hist_out[threadIdx.x]), private_hist[threadIdx.x]);
    //     }
    // }

    __global__ void histogram_gpu(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
        int i = threadIdx.x + blockIdx.x*blockDim.x,
            stride = blockDim.x*gridDim.x;

		atomicAdd(&(hist_out[img_in[i]]), 1);
    }

    __global__ void get_equalized_image(unsigned char *d_img_out, unsigned char *d_img_in, int *d_lut, int d_img_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (d_lut[d_img_in[i]] > 255) {
            d_img_out[i] = 255;
        } else {
            d_img_out[i] = d_lut[d_img_in[i]];
        }
    }

	// Kernel wrapper
    void histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in,
                                int img_size, int nbr_bin) {
        int cdf, min, d, i = 0;
		int *hist_in;
        int *lut;
		
		unsigned char *d_img_in, *d_img_out;
        int *d_hist_out, *d_lut;

        // Host memory allocation
        lut = (int *)malloc(sizeof(int)*nbr_bin);
        hist_in = (int *)malloc(sizeof(int)*nbr_bin);

        // Allocate device memory 
		int padded_size = img_size + (MAX_THREADS_PER_BLOCK - (img_size%MAX_THREADS_PER_BLOCK));		// THIS IS WRONG!!!!!!!!!!!!!!!! (what if we dont need padding (% operation is 0))
		cudaMalloc((void**) &d_img_in,	 sizeof(unsigned char)*padded_size);
        cudaMalloc((void**) &d_hist_out, sizeof(int)*nbr_bin);

        cudaMemset (d_img_in,   0, padded_size);
        cudaMemset (d_hist_out, 0, nbr_bin);
        
		cudaMemcpy(d_img_in, img_in, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 block(MAX_THREADS_PER_BLOCK, 1, 1);
        dim3 grid(GRID_DIM, 1, 1);

        histogram_gpu<<<grid, block>>>(d_hist_out, d_img_in, padded_size, nbr_bin);
        
		cudaDeviceSynchronize();
		checkCudaError("Histogram calculation");

		// Copy calculated histogram back to host 
        cudaMemcpy(hist_in, d_hist_out, sizeof(int)*nbr_bin, cudaMemcpyDeviceToHost);
        
		// Free non-wanted memory 
		cudaFree(d_hist_out);

		// Clean histogram counts added by the padding elements
		// Padding elements are set to 0
        hist_in[0] = hist_in[0] - ((MAX_THREADS_PER_BLOCK - (img_size%MAX_THREADS_PER_BLOCK)));			// THIS IS WRONG!!!!!!!!!!!!!
		/* Construct the LUT by calculating the CDF */
        cdf = 0;
        min = 0;
        while(min == 0){
            min = hist_in[i++];
        }
        d = img_size - min;
        for(i = 0; i < nbr_bin; i ++){
            cdf += hist_in[i];
            lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
            if(lut[i] < 0){
                lut[i] = 0;
            }
        }

		// Device memory allocation
        cudaMalloc((void**) &d_lut,     sizeof(int)*nbr_bin);
        cudaMalloc((void**) &d_img_out, sizeof(unsigned char)*padded_size);

        cudaMemcpy(d_lut, lut, sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);

		// Kernel launch
        get_equalized_image<<<grid, block>>>(d_img_out, d_img_in, d_lut, padded_size);
        
		cudaDeviceSynchronize();
		checkCudaError("Histogram equalization");

        // Copy data back to host memory
        cudaMemcpy(img_out, d_img_out, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_lut);
        cudaFree(d_img_in);
        cudaFree(d_img_out);
        free(lut);
		free(hist_in);

        // Reset the device
        cudaDeviceReset();
    }
}