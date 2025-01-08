#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define THREADS_PER_BLOCK 1024

// Structure of arrays 
typedef struct { 
	float *x;
	float *y;
	float *z;
	float *vx; 
	float *vy;
	float *vz; 
	float *fx;
	float *fy;
	float *fz;
} Body;

/****************************** Helper Functions ******************************/
bool checkCudaError(const char *step) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error in %s: %s\n", step, cudaGetErrorString(err));
		return true;
	}
	return false;
}

void randomizeBodies(Body *bodies, int n) {
  	for (int i = 0; i < n; i++) {
    	bodies->x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

/***************** KERNEL CODE *****************/
__global__ void forceComputeKernel(Body p, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	float dx, dy, dz;
	p.fx[tid] = 0.0f;
	p.fy[tid] = 0.0f;
	p.fz[tid] = 0.0f;

	for (int i = 0; i < n; i++) {
		dx = p.x[i] - p.x[tid];
		dy = p.y[i] - p.y[tid];
		dz = p.z[i] - p.z[tid];
		float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
		float invDist = 1.0f / sqrtf(distSqr);
		float invDist3 = invDist * invDist * invDist;

		p.fx[tid] += dx * invDist3; 
		p.fy[tid] += dy * invDist3; 
		p.fz[tid] += dz * invDist3;
	}
}

__global__ void positionComputeKernel(Body p, float dt, int n) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	p.vx[i] += dt*p.fx[i]; 
	p.vy[i] += dt*p.fy[i]; 
	p.vz[i] += dt*p.fz[i];

	p.x[i] += p.vx[i]*dt;
	p.y[i] += p.vy[i]*dt;
	p.z[i] += p.vz[i]*dt;
}


int main(const int argc, const char** argv) {

  	int nBodies = 30000;
  	if (argc > 1) nBodies = atoi(argv[1]);

  	const float dt = 0.01f; // time step
  	const int nIters = 10;  // simulation iterations

	float totalTime = 0.0f, elapsed_time = 0.0f;
	Body bodies;
	Body d_bodies;

	cudaEvent_t iter_start, iter_end;

	/****************************** Host Memory Allocation ******************************/
	bodies.x  = (float *)malloc(sizeof(float)*nBodies);
	bodies.y  = (float *)malloc(sizeof(float)*nBodies);
	bodies.z  = (float *)malloc(sizeof(float)*nBodies);

	bodies.vx = (float *)malloc(sizeof(float)*nBodies);
	bodies.vy = (float *)malloc(sizeof(float)*nBodies);
	bodies.vz = (float *)malloc(sizeof(float)*nBodies);
	
	bodies.fx = (float *)malloc(sizeof(float)*nBodies);
	bodies.fy = (float *)malloc(sizeof(float)*nBodies);
	bodies.fz = (float *)malloc(sizeof(float)*nBodies);

	cudaEventCreate(&iter_start);
	cudaEventCreate(&iter_end);

  	randomizeBodies(&bodies, nBodies); // Init pos / vel data

	// Set geometry
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	dim3 grid(ceil(nBodies/THREADS_PER_BLOCK), 1, 1);

	/****************************** Device Memory Allocation ******************************/
	cudaMalloc((void **) &d_bodies,    sizeof(Body));
	cudaMalloc((void **) &d_bodies.x,  sizeof(float) * nBodies);
	cudaMalloc((void **) &d_bodies.y,  sizeof(float) * nBodies);
	cudaMalloc((void **) &d_bodies.z,  sizeof(float) * nBodies);

	cudaMalloc((void **) &d_bodies.vx, sizeof(float) * nBodies);
	cudaMalloc((void **) &d_bodies.vy, sizeof(float) * nBodies);
	cudaMalloc((void **) &d_bodies.vz, sizeof(float) * nBodies);

	cudaMalloc((void **) &d_bodies.fx, sizeof(float) * nBodies);
	cudaMalloc((void **) &d_bodies.fy, sizeof(float) * nBodies);
	cudaMalloc((void **) &d_bodies.fz, sizeof(float) * nBodies);

	/****************************** Data Transfers ******************************/
	cudaMemcpy(d_bodies.x,  bodies.x,  sizeof(float) * nBodies, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bodies.y,  bodies.y,  sizeof(float) * nBodies, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bodies.z,  bodies.z,  sizeof(float) * nBodies, cudaMemcpyHostToDevice);

	cudaMemcpy(d_bodies.vx, bodies.vx, sizeof(float) * nBodies, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bodies.vy, bodies.vy, sizeof(float) * nBodies, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bodies.vz, bodies.vz, sizeof(float) * nBodies, cudaMemcpyHostToDevice);

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {
		cudaEventRecord(iter_start, 0);
		
		forceComputeKernel<<<grid, block>>>(d_bodies, nBodies);
		cudaDeviceSynchronize();
		checkCudaError("forceCoputeKernel");

		positionComputeKernel<<<grid, block>>>(d_bodies, dt, nBodies);
		
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
	cudaMemcpy(bodies.x,  d_bodies.x,  sizeof(float) * nBodies, cudaMemcpyDeviceToHost);
	cudaMemcpy(bodies.y,  d_bodies.y,  sizeof(float) * nBodies, cudaMemcpyDeviceToHost);
	cudaMemcpy(bodies.z,  d_bodies.z,  sizeof(float) * nBodies, cudaMemcpyDeviceToHost);

	cudaFree(d_bodies.x);
	cudaFree(d_bodies.y);
	cudaFree(d_bodies.z);
	cudaFree(d_bodies.vx);
	cudaFree(d_bodies.vy);
	cudaFree(d_bodies.vz);
	cudaFree(d_bodies.fx);
	cudaFree(d_bodies.fy);
	cudaFree(d_bodies.fz);

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
		fprintf(fd, "%f\n", bodies.x[i]);
		fprintf(fd, "%f\n", bodies.y[i]);
		fprintf(fd, "%f\n", bodies.z[i]);
	}

	fclose(fd);

	printf("Data written successfully\n");
#endif
	
	free(bodies.x);
	free(bodies.y);
	free(bodies.z);
	free(bodies.vx);
	free(bodies.vy);
	free(bodies.vz);
	free(bodies.fx);
	free(bodies.fy);
	free(bodies.fz);
	
	cudaDeviceReset();
}
