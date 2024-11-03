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
#include "kmeans.h"

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
    int      i, j, k, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta, min_dist, dist;          /* % of objects change their clusters */
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
            #pragma omp for private(j, index, k, min_dist, dist) reduction(+:delta, newClusterSize[:numClusters], newClusters[:numClusters*numCoords]) schedule(auto)
            for (i=0; i<numObjs; i++) {
                /* find the array index of nestest cluster center */

                index = 0;
                min_dist=0.0;
                for (k=0; k<numCoords; k++)
                    min_dist += (objects[i][k]-clusters[0][k]) * (objects[i][k]-clusters[0][k]);

                for (j=1; j<numClusters; j++) {
                    dist=0.0;
                    for (k=0; k<numCoords; k++)
                        dist += (objects[i][k]-clusters[j][k]) * (objects[i][k]-clusters[j][k]);

                    /* no need square root */
                    if (dist < min_dist) { /* find the min and its array index */
                        min_dist = dist;
                        index    = j;
                    }
                }

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