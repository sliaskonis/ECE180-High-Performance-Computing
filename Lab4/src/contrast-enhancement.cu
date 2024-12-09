#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "hist-equ.h"
#include "colours.h"

PGM_IMG contrast_enhancement_cpu(PGM_IMG img_in)
{
    PGM_IMG result;
    int histo[256];
    clock_t start, hist, hist_equ;
    double total_elapsed_time, hist_time, hist_equ_time;

    result.w = img_in.w;
    result.h = img_in.h;
    cudaHostAlloc((void**) &result.img, result.w * result.h * sizeof(unsigned char), cudaHostAllocDefault);

    start = clock();
    histogram(histo, img_in.img, img_in.h * img_in.w, 256);
    hist = clock();

    histogram_equalization_cpu(result.img, img_in.img, histo, result.w*result.h, 256);
    hist_equ = clock();

    // Print timing results
    total_elapsed_time = (double) (hist_equ - start) / CLOCKS_PER_SEC;
    hist_time = (double) (hist - start) / CLOCKS_PER_SEC;
    hist_equ_time = (double) (hist_equ - hist) / CLOCKS_PER_SEC;

    printf( GRN "Total CPU time: %fsec, consists of:\n"       RESET, total_elapsed_time);
    printf( MAG "\t%f (histogram calculation)\n"              RESET, hist_time);
    printf( MAG "\t%f (histogram equalization calculation)\n" RESET, hist_equ_time);

    return result;
}
