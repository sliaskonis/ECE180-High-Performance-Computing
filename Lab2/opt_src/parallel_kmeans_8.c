/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c  (sequential version)                        */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h> 
#include "kmeans.h"

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */


__inline static 
float euclid_dist_2(int numdims, float *restrict coord1, float *restrict coord2) {
    float ans = 0.0f;
    int i;

    // Initialize SIMD sum to 0
    __m128 sum = _mm_setzero_ps();

    for (i = 0; i <= numdims - 4; i += 4) {
        __m128 vec1 = _mm_loadu_ps(&coord1[i]);
        __m128 vec2 = _mm_loadu_ps(&coord2[i]);
        __m128 diff = _mm_sub_ps(vec1, vec2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    // Perform horizontal addition on sum
    float tmp[4];
    _mm_storeu_ps(tmp, sum);
    ans = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // Process remaining elements
    for (; i < numdims; i++) {
        float diff = coord1[i] - coord2[i];
        ans += diff * diff;
    }

    return ans;
}


/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
int seq_kmeans(float **objects,      /* in: [numObjs][numCoords] */
               int     numCoords,    /* no. features */
               int     numObjs,      /* no. objects */
               int     numClusters,  /* no. clusters */
               float   threshold,    /* % objects change membership */
               int    *membership,   /* out: [numObjs] */
               float **clusters)     /* out: [numClusters][numCoords] */

{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  *newClusters;    /* [numClusters][numCoords] */

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters = (float *) calloc(numClusters*numCoords, sizeof(float));
    assert(newClusters != NULL);

    do {
        delta = 0.0;
	    #pragma omp parallel
        { 
            #pragma omp for private(j, index) reduction(+:delta, newClusterSize[:numClusters], newClusters[:numClusters*numCoords]) schedule(auto)
            for (i=0; i<numObjs; i++) {
                /* find the array index of nestest cluster center */
                index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                            clusters);

                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta += 1.0;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster center : sum of objects located within */
                newClusterSize[index]++;
                for (j=0; j<numCoords; j++) {
                    newClusters[index*numCoords+j] += objects[i][j];
                }
            }

            /* average the sum and replace old cluster center with newClusters */
            #pragma omp for private(j) schedule(auto)
            for (i=0; i<numClusters; i++) {
                for (j=0; j<numCoords; j++) {
                    if (newClusterSize[i] > 0)
                        clusters[i][j] = newClusters[i*numCoords+j] / newClusterSize[i];
                    newClusters[i*numCoords+j] = 0.0;   /* set back to 0 */
                }
                newClusterSize[i] = 0;   /* set back to 0 */
            }
        }            
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    free(newClusters);
    free(newClusterSize);

    return 1;
}