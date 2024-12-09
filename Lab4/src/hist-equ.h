#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;

#ifdef __cplusplus
extern "C" {
#endif
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);
void free_pgm_gpu(PGM_IMG img);

// Histogram equalization function implemntation (host & device implementation)
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization_cpu(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin);
void histogram_gpu(unsigned char *img, int img_size, int nbr_bin);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_cpu(PGM_IMG img_in);
#ifdef __cplusplus
}
#endif
#endif
