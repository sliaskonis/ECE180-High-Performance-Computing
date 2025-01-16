#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define THREADS_PER_BLOCK 1024

typedef struct { 
	float4 *pos; 
	float4 *vel;
} Body;

/****************************** Helper Functions ******************************/
void checkCudaError(const char *step) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error in %s: %s\n", step, cudaGetErrorString(err));
		cudaDeviceReset();
		exit(1);
	}
}

void randomizeBodies(Body *bodies, int n) {
  	for (int i = 0; i < n; i++) {
    	bodies->pos[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->pos[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->pos[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vel[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vel[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vel[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  	}
}

/***************** KERNEL CODE *****************/
__global__ void bodyForce(Body p, float dt, int tiles, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	signed int tile;
	
	float4 diff;
	float distSqr, invDist, invDist3;
	float Fx = 0.0f;
	float Fy = 0.0f;
	float Fz = 0.0f;

	__shared__ float4 body_coordinates_pos[THREADS_PER_BLOCK];
	float4 curr_pos = p.pos[tid];
	
	for (tile = 0; tile < tiles-1; tile++) {
		body_coordinates_pos[threadIdx.x] = p.pos[threadIdx.x + tile*blockDim.x];

		__syncthreads();
		#pragma unroll 16
		for (signed int i = 0; i < THREADS_PER_BLOCK; i++) {
			diff = body_coordinates_pos[i] - curr_pos;
			distSqr = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + SOFTENING;
			invDist = rsqrtf(distSqr);
			invDist3 = invDist * invDist * invDist;

			Fx += diff.x * invDist3; 
			Fy += diff.y * invDist3; 
			Fz += diff.z * invDist3;
		}
		__syncthreads();
	}

	// Bring last tile into shared memory;
	body_coordinates_pos[threadIdx.x] = p.pos[threadIdx.x + (tiles-1)*blockDim.x];
	__syncthreads();

	int last_bodies = (n%THREADS_PER_BLOCK == 0) ? THREADS_PER_BLOCK : n&(THREADS_PER_BLOCK-1);

	#pragma unroll 16
	for (signed int i = 0; i < last_bodies; i++) {
		diff = body_coordinates_pos[i] - curr_pos;
		distSqr = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + SOFTENING;
		invDist = rsqrtf(distSqr);
		invDist3 = invDist * invDist * invDist;

		Fx += diff.x * invDist3; 
		Fy += diff.y * invDist3; 
		Fz += diff.z * invDist3;
	}

    p.vel[tid].x += dt*Fx;
	p.vel[tid].y += dt*Fy;
	p.vel[tid].z += dt*Fz;
}

__global__ void calculatePositions(Body p, float dt, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= n) {
		return;
	}

	p.pos[tid].x += p.vel[tid].x*dt;
	p.pos[tid].y += p.vel[tid].y*dt;
	p.pos[tid].z += p.vel[tid].z*dt;
}

int main(const int argc, const char** argv) {

  	int nBodies = 30000;
  	if (argc > 1) nBodies = atoi(argv[1]);

  	const float dt = 0.01f; // time step
  	const int nIters = 10;  // simulation iterations

	float totalTime = 0.0f, elapsed_time = 0.0f;
	Body bodies, d_bodies;

	cudaEvent_t iter_start, iter_end;

	cudaEventCreate(&iter_start);
	cudaEventCreate(&iter_end);

	/****************************** Host memory allocation ******************************/
	int bytes = sizeof(float4)*nBodies;
	bodies.pos = (float4 *)malloc(bytes);
	bodies.vel = (float4 *)malloc(bytes);

  	randomizeBodies(&bodies, nBodies); // Init pos / vel data

	// Set geometry
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	dim3 grid((int)(ceil((float)nBodies/THREADS_PER_BLOCK)), 1, 1);
	int tiles = (int)(ceil((float)nBodies/THREADS_PER_BLOCK));

	/****************************** Data transfers ******************************/
	cudaMalloc((void **) &d_bodies.pos, bytes);
	cudaMalloc((void **) &d_bodies.vel, bytes);

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {
		cudaEventRecord(iter_start, 0);

		// In the first iteration both initial coordinates and velocity needs to be copied to device
		if (iter == 1) {
			cudaMemcpy(d_bodies.pos,  bodies.pos,  bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bodies.vel,  bodies.vel, bytes, cudaMemcpyHostToDevice);
		}

		bodyForce<<<grid, block>>>(d_bodies, dt, tiles, nBodies);
		checkCudaError("bodyForce");
        cudaDeviceSynchronize();

		calculatePositions<<<grid, block>>>(d_bodies, dt, nBodies);
		cudaDeviceSynchronize();

		// Send final coordinates back to host
		if (iter == nIters) {
			cudaMemcpy(bodies.pos, d_bodies.pos, bytes, cudaMemcpyDeviceToHost);
		}

		/****************************** Save Final Coordinates ******************************/
		#ifdef SAVE_FINAL_COORDINATES
		if (iter == 2) {
			// Copy coordinates back to host for checking
			cudaMemcpy(bodies.pos, d_bodies.pos, bytes, cudaMemcpyDeviceToHost);

			char filename[256];
		
			sprintf(filename, "cuda_coordinates_%d.txt", nBodies);
		
			printf("Writing final coordinates to %s\n", filename);
			FILE *fd = fopen(filename, "w");
		
			if (!fd) {
				perror("Failed opening file");
				return -1;
			}
		
			for (int i = 0; i < nBodies; i++) {
				fprintf(fd, "%f\n", bodies.pos[i].x);
				fprintf(fd, "%f\n", bodies.pos[i].y);
				fprintf(fd, "%f\n", bodies.pos[i].z);
			}
		
			fclose(fd);
		
			printf("Data written successfully\n");
		}		
		#endif

		cudaEventRecord(iter_end, 0);
		cudaEventSynchronize(iter_end);

		cudaEventElapsedTime(&elapsed_time, iter_start, iter_end);
		if (iter > 1) { // First iter is warm up
			totalTime += elapsed_time/1000.0f;
		}
		printf("Iteration %d: %.3f seconds\n", iter, elapsed_time/1000.0f);
  	}

  	float avgTime = totalTime / (float)(nIters-1);

  	printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
	printf("Total time: %.3f\n", totalTime);

	/****************************** Cleanup ******************************/
	// Device
	cudaFree(d_bodies.pos);
	cudaFree(d_bodies.vel);
	
	// Host
	free(bodies.pos);
	free(bodies.vel);	

	cudaDeviceReset();
}