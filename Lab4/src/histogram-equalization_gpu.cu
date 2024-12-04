extern "C" {
    #include <stdio.h>
    #include <cuda_runtime.h>
    #include "hist-equ.h"

    #define MAX_THREADS_PER_BLOCK 1024
    #define GRID_DIM ceil((float)img_size/MAX_THREADS_PER_BLOCK)

    __global__ void get_equalized_image(unsigned char *d_img_out, unsigned char *d_img_in, int *d_lut, int d_img_size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (d_lut[d_img_in[i]] > 255) {
            d_img_out[i] = 255;
        } else {
            d_img_out[i] = d_lut[d_img_in[i]];
        }
    }

    // TODO: Add padding to the input image
    void histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in,
                                int *hist_in, int img_size, int nbr_bin) {
        int *lut = (int *)malloc(sizeof(int)*nbr_bin);
        int cdf, min, d;
        int *d_lut;
        unsigned char *d_img_in, *d_img_out;
        int i = 0;

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

        // Allocate device memory
        cudaMalloc((void**) &d_lut, sizeof(int)*nbr_bin);
        cudaMalloc((void**) &d_img_in, sizeof(unsigned char)*img_size);
        cudaMalloc((void**) &d_img_out, sizeof(unsigned char)*img_size);

        // Copy data to device memory
        cudaMemcpy(d_lut, lut, sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);
        cudaMemcpy(d_img_in, img_in, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 block(MAX_THREADS_PER_BLOCK, 1, 1);
        dim3 grid(GRID_DIM, 1, 1);

        get_equalized_image<<<grid, block>>>(d_img_out, d_img_in, d_lut, img_size);

        // Copy data back to host memory
        cudaMemcpy(img_out, d_img_out, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_lut);
        cudaFree(d_img_in);
        cudaFree(d_img_out);
        free(lut);

        // Reset the device
        cudaDeviceReset();
    }
}