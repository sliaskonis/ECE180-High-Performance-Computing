#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "hist-equ.h"
#include "colours.h"

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);
void run_gpu_gray_test(PGM_IMG img_in, char *out_filename);

bool checkCudaError(const char *step) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in %s: %s\n", step, cudaGetErrorString(err));
        return true;
    }
    return false;
}

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;

	if (argc != 4) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}

    img_ibuf_g = read_pgm(argv[1]);

    // printf(YEL "Running contrast enhancement for gray-scale images.\n" RESET);

    // run_cpu_gray_test(img_ibuf_g, argv[2]);

    printf(YEL "\nRunning contrast enhancement for gray-scale images on gpu.\n" RESET);

    run_gpu_gray_test(img_ibuf_g, argv[3]);
    
    free_pgm(img_ibuf_g);

    // Reset the device
    cudaDeviceReset();

    return 0;
}

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    PGM_IMG img_obuf;

    printf(YEL "Starting CPU processing...\n" RESET);
    img_obuf = contrast_enhancement_cpu(img_in);
    
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
}

void run_gpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    printf(YEL "Starting GPU processing...\n" RESET);

    histogram_gpu(img_in.img, img_in.w*img_in.h, 256);   
    
    write_pgm(img_in, out_filename);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];

    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }

    if (fscanf(in_file, "%s", sbuf) != 1) {
        fprintf(stderr, "Error reading magic number\n");
        exit(EXIT_FAILURE);
    }
    if (fscanf(in_file, "%d", &result.w) != 1) {
        fprintf(stderr, "Error reading width\n");
        exit(EXIT_FAILURE);
    }
    if (fscanf(in_file, "%d", &result.h) != 1) {
        fprintf(stderr, "Error reading height\n");
        exit(EXIT_FAILURE);
    }
    if (fscanf(in_file, "%d", &v_max) != 1) {
        fprintf(stderr, "Error reading max value\n");
        exit(EXIT_FAILURE);
    }

    printf("Image size: %d x %d\n", result.w, result.h);

    // TODO: try different flags and monitor behaviour
    // cudaHostAlloc((void**) &result.img, result.w * result.h * sizeof(unsigned char), cudaHostAllocMapped);
    // cudaHostAlloc((void**) &result.img, result.w * result.h * sizeof(unsigned char), cudaHostAllocWriteCombined); // need mapped too in our case

    cudaHostAlloc((void**) &result.img, result.w * result.h * sizeof(unsigned char), cudaHostAllocMapped | cudaHostAllocWriteCombined);
    
    checkCudaError("cudaHostAlloc");

    if (fread(result.img, sizeof(unsigned char), result.w * result.h, in_file) != (size_t)(result.w * result.h)) {
        fprintf(stderr, "Error reading image data\n");
        exit(EXIT_FAILURE);
    }
    fclose(in_file);

    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;

    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    cudaFreeHost(img.img);
    checkCudaError("cudaFreeHost");
}

