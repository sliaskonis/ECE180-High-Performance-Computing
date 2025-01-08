#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define THREADS_PER_BLOCK 1024

typedef struct { float x, y, z, vx, vy, vz;} Body;

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
  	for (int i = 0; i < n; i++) {
    	data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  	}
}

/***************** KERNEL CODE *****************/
__global__ void bodyForce(Body *p, float dt, int tiles, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int tile;

	float dx, dy, dz;
	float distSqr, invDist, invDist3;
	float Fx = 0.0f;
	float Fy = 0.0f;
	float Fz = 0.0f;

	__shared__ Body private_bodies[THREADS_PER_BLOCK];
	Body curr_body = p[tid];

	for (tile = 0; tile < tiles; tile++) {
		private_bodies[threadIdx.x] = p[threadIdx.x + tile * blockDim.x];
		__syncthreads();
		for (int i = 0; i < THREADS_PER_BLOCK; i++) {
			dx = private_bodies[i].x - curr_body.x;
			dy = private_bodies[i].y - curr_body.y;
			dz = private_bodies[i].z - curr_body.z;
			distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
			invDist = 1.0f / sqrtf(distSqr);
			invDist3 = invDist * invDist * invDist;

			Fx += dx * invDist3;
			Fy += dy * invDist3; 
			Fz += dz * invDist3;
		}
		__syncthreads();
	}

	// // Load last tile into shared memory
	// private_bodies[threadIdx.x] = p[threadIdx.x + (tiles-1) * blockDim.x];
	// __syncthreads();

	// int last_bodies = (n%THREADS_PER_BLOCK == 0) ? THREADS_PER_BLOCK : (THREADS_PER_BLOCK - n%THREADS_PER_BLOCK);

	// for (int j = 0; j < last_bodies; j++) {
	// 	dx = private_bodies[j].x - curr_body.x;
	// 	dy = private_bodies[j].y - curr_body.y;
	// 	dz = private_bodies[j].z - curr_body.z;
	// 	distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
	// 	invDist = 1.0f / sqrtf(distSqr);
	// 	invDist3 = invDist * invDist * invDist;

	// 	Fx += dx * invDist3;
	// 	Fy += dy * invDist3; 
	// 	Fz += dz * invDist3;
	// }

    curr_body.vx += dt*Fx;
	curr_body.vy += dt*Fy;
	curr_body.vz += dt*Fz;

	// Update global memory
	p[tid] = curr_body;
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

  	randomizeBodies(buf, 6*nBodies); // Init pos / vel data

	// Set geometry
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	dim3 grid((int)(ceil(nBodies/THREADS_PER_BLOCK)), 1, 1);
	int tiles = (int)(ceil(nBodies/THREADS_PER_BLOCK));

	/****************************** Data transfers ******************************/
	cudaMalloc((void **) &d_buf, bytes);

	d_p = (Body*)d_buf;

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {
		cudaEventRecord(iter_start, 0);

		// Tranfer new coordinates back to device for next computations
		cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
		
		bodyForce<<<grid, block>>>(d_p, dt, tiles, nBodies);
		checkCudaError("bodyForce");
        cudaDeviceSynchronize();

        // Transfer data back to host in order to compute new coordinates
	    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

		// Calculate new coordinates
        for (int i = 0 ; i < nBodies; i++) {
            p[i].x += p[i].vx*dt;
            p[i].y += p[i].vy*dt;
            p[i].z += p[i].vz*dt;
        }

        cudaEventRecord(iter_end, 0);
		cudaEventSynchronize(iter_end);

		cudaEventElapsedTime(&elapsed_time, iter_start, iter_end);
    	if (iter > 1) { // First iter is warm up
      		totalTime += elapsed_time/1000.0f;
    	}
    	printf("Iteration %d: %.3f seconds\n", iter, elapsed_time/1000.0f);
  	}

  	float avgTime = totalTime / (float)(nIters-1);

	/****************************** Data transfers ******************************/
	cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

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
	
	/****************************** Cleanup ******************************/
	cudaFree(d_buf);
	free(buf);
	
	cudaDeviceReset();
}