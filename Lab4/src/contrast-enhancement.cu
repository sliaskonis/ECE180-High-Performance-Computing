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
    clock_t start, hist, end;
    double time_taken;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    start = clock();
    histogram(histo, img_in.img, img_in.h * img_in.w, 256);
    hist = clock();

    histogram_equalization_cpu(result.img, img_in.img, histo, result.w*result.h, 256);
    end = clock();

    time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf(GRN "Total CPU time: %fsec, consists of:\n" RESET, time_taken);

    time_taken = (double)(hist - start) / CLOCKS_PER_SEC;
    printf(MAG"\t%f (histogram calculation)\n" RESET, time_taken);
    
    time_taken = (double)(end - hist) / CLOCKS_PER_SEC;
    printf(MAG"\t%f\n" RESET, time_taken);

    return result;
}
