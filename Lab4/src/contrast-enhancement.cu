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

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(histo, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization_cpu(result.img, img_in.img, histo, result.w*result.h, 256);

    return result;
}
