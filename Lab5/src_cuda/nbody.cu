#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */

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
__global__ void forceComputeKernel(Body *p, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	float dx, dy, dz;
	float distSqr, invDist, invDist3;
	p[tid].Fx = 0.0f;
	p[tid].Fy = 0.0f;
	p[tid].Fz = 0.0f;

	for (int i = 0; i < n; i++) {
		dx = p[i].x - p[tid].x;
		dy = p[i].y - p[tid].y;
		dz = p[i].z - p[tid].z;
		float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
		float invDist = 1.0f / sqrtf(distSqr);
		float invDist3 = invDist * invDist * invDist;

		p[tid].Fx += dx * invDist3; p[tid].Fy += dy * invDist3; p[tid].Fz += dz * invDist3;
	}
}

__global__ void positionComputeKernel(Body *p, float dt, int n) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	p[i].vx += dt*p[i].Fx; p[i].vy += dt*p[i].Fy; p[i].vz += dt*p[i].Fz;

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
	dim3 block(1024, 1, 1);
	dim3 grid(ceil(nBodies/1024), 1, 1);

	/****************************** Data transfers ******************************/
	cudaMalloc((void **) &d_buf, bytes);
	cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
	d_p = (Body*)d_buf;

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {
		cudaEventRecord(iter_start, 0);
		
		forceComputeKernel<<<grid, block>>>(d_p, nBodies);
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
	printf("Writing final coordinates to cuda_nbody.txt\n");
	FILE *fd = fopen("cuda_nbody.txt", "w");

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
