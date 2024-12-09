extern "C" {
    #include <stdio.h>
    #include <cuda_runtime.h>
    #include <time.h>
    #include "hist-equ.h"
    #include "colours.h"

    #define MAX_THREADS_PER_BLOCK 1024
    #define BLOCK_SIZE 256
    #define CFACTOR 10
    #define STRIDE 200
    #define GRID_DIM_1 ceil(((float)img_size/BLOCK_SIZE)/CFACTOR)
    #define GRID_DIM_2 ceil(((float)img_size/MAX_THREADS_PER_BLOCK)/STRIDE)

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
    __global__ void histogram_calc(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
        __shared__ int private_hist[256];

        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int it = 0, accum = 0, prev_pixel_val = -1;

        it = i*CFACTOR;

        if (threadIdx.x < 256) {
            private_hist[threadIdx.x] = 0;
        }
        __syncthreads();

        while (it < min(img_size, (i+1)*CFACTOR)) {
            int pixel_val = img_in[it];
            if (pixel_val == prev_pixel_val) {
                accum++;
            }
            else{
                if (accum > 0) {
                    atomicAdd(&(private_hist[prev_pixel_val]), accum);
                }
                accum = 1;
                prev_pixel_val = pixel_val;
            }
            it++;
        }

        if (accum > 0) {
            atomicAdd(&(private_hist[prev_pixel_val]), accum);
        }
        __syncthreads();

        if (threadIdx.x < 256) {
            atomicAdd(&(hist_out[threadIdx.x]), private_hist[threadIdx.x]);
        }
    }

    // CDF kernel
    __global__ void cdf_calc(int *d_hist, int img_size, int nbr_bin) {
        int min = 0, d, cdf = 0, idx = 0;

        __shared__ int priv_hist[256];

        if (threadIdx.x < 256) {
            priv_hist[threadIdx.x] = d_hist[threadIdx.x];
        }
        __syncthreads();

        while (min == 0) {
            min = priv_hist[idx++];
        }
        d = img_size - min;

        for (int i = 0; i <= threadIdx.x; i++) {
            cdf += priv_hist[i];
        }

        d_hist[threadIdx.x] = max((int)(((float)cdf - min)*255/d + 0.5), 0);
    }


    // Histogram equalization application: privatization, interleaved partitioning of threads -> coalesced memory accesses
    __global__ void histogram_equ(unsigned char *d_img_in, int *d_lut, int img_size) {
        int tid = threadIdx.x + blockIdx.x*blockDim.x,
            stride = blockDim.x * gridDim.x;
        __shared__ int priv_lut[256];

        if (threadIdx.x < 256) {
            priv_lut[threadIdx.x] = d_lut[threadIdx.x];
        }
        __syncthreads();

        while (tid < img_size) {
            d_img_in[tid] = (unsigned char)priv_lut[d_img_in[tid]];
            tid += stride;
        }
    }

	// Kernel wrapper
    void histogram_gpu(unsigned char *img_in,
                                int img_size, int nbr_bin) {
        float elapsed_time;
		unsigned char *d_img_in;
        int *d_hist_out;

        cudaEvent_t gpu_start, gpu_stop, memory_transfers, hist_kernel, cdf_kernel, hist_equ_kernel_end;
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_stop);
        cudaEventCreate(&memory_transfers);
        cudaEventCreate(&hist_kernel);
        cudaEventCreate(&cdf_kernel);
        cudaEventCreate(&hist_equ_kernel_end);

        dim3 block(BLOCK_SIZE, 1, 1);
        dim3 grid(GRID_DIM_1, 1, 1);

        cudaEventRecord(gpu_start, 0);

        /************************* Device Memory Allocation *************************/
		cudaMalloc((void**) &d_img_in,	 sizeof(unsigned char)*img_size);
        cudaMalloc((void**) &d_hist_out, sizeof(int)*nbr_bin);

        cudaMemset (d_img_in,   0, sizeof(unsigned char)*img_size);
        cudaMemset (d_hist_out, 0, sizeof(int)*nbr_bin);
		cudaMemcpy(d_img_in, img_in, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
                                
        cudaEventRecord(memory_transfers, 0);

        /************************* Histogram calculation kernel launch *************************/
        histogram_calc<<<grid, block>>>(d_hist_out, d_img_in, img_size, nbr_bin);

        cudaEventRecord(hist_kernel, 0);
        cudaEventSynchronize(hist_kernel);

		checkCudaError("Histogram calculation");

        /************************* CDF calculation kernel launch *************************/
        cdf_calc<<<1, 256>>>(d_hist_out, img_size, nbr_bin);

        cudaEventRecord(cdf_kernel, 0);
        cudaEventSynchronize(cdf_kernel);

        checkCudaError("Cdf calculation");
        
        dim3 block2(MAX_THREADS_PER_BLOCK, 1, 1);
        dim3 grid2(GRID_DIM_2, 1, 1);

        /************************* Histogram equalization kernel launch *************************/
        histogram_equ<<<grid2, block2>>>(d_img_in, d_hist_out, img_size);

        cudaEventRecord(hist_equ_kernel_end, 0);
        cudaEventSynchronize(hist_equ_kernel_end);

        checkCudaError("Histogram equalization");

        // Copy img back to host
        cudaMemcpy(img_in, d_img_in, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);
        
        // Free non-wanted memory
        cudaFree(d_img_in);
        cudaFree(d_hist_out);

        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);

        // Calculate elapsed time for all events
        cudaEventElapsedTime(&elapsed_time, gpu_start, gpu_stop);
        printf( GRN "Total GPU time: %fsec, consists of:\n" RESET, elapsed_time/1000);

        cudaEventElapsedTime(&elapsed_time, gpu_start, memory_transfers);
        printf(MAG"\t%f (memory transfers 1)\n" RESET, elapsed_time/1000);

        cudaEventElapsedTime(&elapsed_time, memory_transfers, hist_kernel);
        printf(MAG"\t%f (histogram kernel)\n" RESET, elapsed_time/1000);

        cudaEventElapsedTime(&elapsed_time, hist_kernel, cdf_kernel);
        printf(MAG"\t%f (cdf calculation)\n" RESET, elapsed_time/1000);

        cudaEventElapsedTime(&elapsed_time, cdf_kernel, hist_equ_kernel_end);
        printf(MAG"\t%f (histogram equalization kernel)\n" RESET, elapsed_time/1000);

        cudaEventElapsedTime(&elapsed_time, hist_equ_kernel_end, gpu_stop);
        printf(MAG"\t%f (memory transfers + cleanup)\n" RESET, elapsed_time/1000);

        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_stop);
        cudaEventDestroy(memory_transfers);
        cudaEventDestroy(hist_kernel);
        cudaEventDestroy(cdf_kernel);
        cudaEventDestroy(hist_equ_kernel_end);

    }
}