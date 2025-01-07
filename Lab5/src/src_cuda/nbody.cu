#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define THREADS_PER_BLOCK 1024

typedef struct { float x, y, z, vx, vy, vz, Fx, Fy, Fz; } Body;

/****************************** Helper Functions ******************************/
bool checkCudaError(const char *step) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error in %s: %s\n", step, cudaGetErrorString(err));
		return true;
	}
	return false;
}

void randomizeBodies(float *data, int n) {
  	for (int i = 0; i < n; i+=9) {
    	data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i+1] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i+2] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i+3] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i+4] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data[i+5] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  	}
}

/***************** KERNEL CODE *****************/
__global__ void forceComputeKernel(Body *p, int n, int tiles) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	float dx, dy, dz;
    float distSqr __attribute__((unused));
    float invDist __attribute__((unused));
    float invDist3 __attribute__((unused));
    int tile;
	Body curr_body;
    __shared__ Body private_bodies[THREADS_PER_BLOCK];

	p[tid].Fx = 0.0f;
	p[tid].Fy = 0.0f;
	p[tid].Fz = 0.0f;

	curr_body = p[tid];

    // Load tile bodies into shared memory
    // Let each body calculate its interaction with the other bodies from shared memory
    // Then load next bodies into shared memory
    for (tile = 0; tile < tiles; tile++) {
        // Load tile to memory
		private_bodies[threadIdx.x] = p[threadIdx.x + tile*blockDim.x];
        __syncthreads();
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            dx = private_bodies[i].x - curr_body.x;
            dy = private_bodies[i].y - curr_body.y;
            dz = private_bodies[i].z - curr_body.z;
            distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            curr_body.Fx += dx * invDist3; 
			curr_body.Fy += dy * invDist3; 
			curr_body.Fz += dz * invDist3;
        }
		__syncthreads();
	}
	p[tid] = curr_body;
}

__global__ void positionComputeKernel(Body *p, float dt, int n) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	p[i].vx += dt*p[i].Fx; 
    p[i].vy += dt*p[i].Fy; 
    p[i].vz += dt*p[i].Fz;

	p[i].x += p[i].vx*dt;
	p[i].y += p[i].vy*dt;
	p[i].z += p[i].vz*dt;
}


int main(const int argc, const char** argv) {

  	int nBodies = 30000;
  	if (argc > 1) nBodies = atoi(argv[1]);

  	const float dt = 0.01f; // time step
  	const int nIters = 10;  // simulation iterations

  	int bytes = nBodies*sizeof(Body);
	float totalTime = 0.0f, elapsed_time = 0.0f;
	float *buf = (float*)malloc(bytes);
	float *d_buf;
	Body *d_p, *p = (Body*)buf;
	cudaEvent_t iter_start, iter_end;

	cudaEventCreate(&iter_start);
	cudaEventCreate(&iter_end);

  	randomizeBodies(buf, 9*nBodies); // Init pos / vel data

	// Set geometry
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	dim3 grid(ceil(nBodies/THREADS_PER_BLOCK), 1, 1);
	int tiles = (int)ceil(nBodies/THREADS_PER_BLOCK);

	/****************************** Data transfers ******************************/
	cudaMalloc((void **) &d_buf, bytes);
	cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
	d_p = (Body*)d_buf;

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {
		cudaEventRecord(iter_start, 0);
		
		forceComputeKernel<<<grid, block>>>(d_p, nBodies, tiles);
		cudaDeviceSynchronize();
		checkCudaError("forceCoputeKernel");

		positionComputeKernel<<<grid, block>>>(d_p, dt, nBodies);
		
		cudaEventRecord(iter_end, 0);
		cudaEventSynchronize(iter_end);
		checkCudaError("positionCoputeKernel");

		cudaEventElapsedTime(&elapsed_time, iter_start, iter_end);

    	if (iter > 1) { // First iter is warm up
      		totalTime += elapsed_time/1000.0f;
    	}
    	printf("Iteration %d: %.3f seconds\n", iter, elapsed_time/1000.0f);
  	}

  	float avgTime = totalTime / (float)(nIters-1);

	/****************************** Data transfers ******************************/
	cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_buf);

  	printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
	printf("Total time: %.3f\n", totalTime);

#ifdef SAVE_FINAL_COORDINATES
	/****************************** Save Final Coordinates ******************************/
	char filename[256];

	sprintf(filename, "cuda_coordinates_%d.txt", nBodies);

	printf("Writing final coordinates to %s\n", filename);
	FILE *fd = fopen(filename, "w");

	if (!fd) {
		perror("Failed opening file");
		return -1;
	}

	for (int i = 0; i < nBodies; i++) {
		fprintf(fd, "%f\n", p[i].x);
		fprintf(fd, "%f\n", p[i].y);
		fprintf(fd, "%f\n", p[i].z);
	}

	fclose(fd);

	printf("Data written successfully\n");
#endif
	
	free(buf);
	
	cudaDeviceReset();
}