#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define THREADS_PER_BLOCK 1024

typedef struct { float *x, *y, *z, *vx, *vy, *vz;} Body;

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
    	bodies->x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    	bodies->vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  	}
}

/***************** KERNEL CODE *****************/
__global__ void bodyForce(Body p, float dt, int tiles, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int tile;
	
	float dx, dy, dz;
	float distSqr, invDist, invDist3;
	float Fx = 0.0f;
	float Fy = 0.0f;
	float Fz = 0.0f;

	__shared__ float body_coordinates_x[THREADS_PER_BLOCK];
	__shared__ float body_coordinates_y[THREADS_PER_BLOCK];
	__shared__ float body_coordinates_z[THREADS_PER_BLOCK];
	float curr_x = p.x[tid];
	float curr_y = p.y[tid];
	float curr_z = p.z[tid];
	
	for (tile = 0; tile < tiles-1; tile++) {
		body_coordinates_x[threadIdx.x] = p.x[threadIdx.x + tile*blockDim.x];
		body_coordinates_y[threadIdx.x] = p.y[threadIdx.x + tile*blockDim.x];
		body_coordinates_z[threadIdx.x] = p.z[threadIdx.x + tile*blockDim.x];

		__syncthreads();
		#pragma unroll 16
		for (int i = 0; i < THREADS_PER_BLOCK; i++) {
			dx = body_coordinates_x[i] - curr_x;
			dy = body_coordinates_y[i] - curr_y;
			dz = body_coordinates_z[i] - curr_z;
			distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
			invDist = rsqrtf(distSqr);
			invDist3 = invDist * invDist * invDist;

			Fx += dx * invDist3; 
			Fy += dy * invDist3; 
			Fz += dz * invDist3;
		}
		__syncthreads();
	}

	// Bring last tile into shared memory;
	body_coordinates_x[threadIdx.x] = p.x[threadIdx.x + (tiles-1)*blockDim.x];
	body_coordinates_y[threadIdx.x] = p.y[threadIdx.x + (tiles-1)*blockDim.x];
	body_coordinates_z[threadIdx.x] = p.z[threadIdx.x + (tiles-1)*blockDim.x];
	__syncthreads();

	int last_bodies = (n%THREADS_PER_BLOCK == 0) ? THREADS_PER_BLOCK : n%THREADS_PER_BLOCK;

	#pragma unroll 16
	for (int i = 0; i < last_bodies; i++) {
		dx = body_coordinates_x[i] - curr_x;
		dy = body_coordinates_y[i] - curr_y;
		dz = body_coordinates_z[i] - curr_z;
		distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
		invDist = rsqrtf(distSqr);
		invDist3 = invDist * invDist * invDist;

		Fx += dx * invDist3; 
		Fy += dy * invDist3; 
		Fz += dz * invDist3;
	}

    p.vx[tid] += dt*Fx;
	p.vy[tid] += dt*Fy;
	p.vz[tid] += dt*Fz;
}

__global__ void calculatePositions(Body p, float dt, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= n) {
		return;
	}

	p.x[tid] += p.vx[tid]*dt;
	p.y[tid] += p.vy[tid]*dt;
	p.z[tid] += p.vz[tid]*dt;
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
	int bytes = sizeof(float)*nBodies;
	bodies.x = (float *)malloc(bytes);
	bodies.y = (float *)malloc(bytes);
	bodies.z = (float *)malloc(bytes);
	bodies.vx = (float *)malloc(bytes);
	bodies.vy = (float *)malloc(bytes);
	bodies.vz = (float *)malloc(bytes);

  	randomizeBodies(&bodies, nBodies); // Init pos / vel data

	// Set geometry
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	dim3 grid((int)(ceil((float)nBodies/THREADS_PER_BLOCK)), 1, 1);
	int tiles = (int)(ceil((float)nBodies/THREADS_PER_BLOCK));

	/****************************** Data transfers ******************************/
	cudaMalloc((void **) &d_bodies.x, bytes);
	cudaMalloc((void **) &d_bodies.y, bytes);
	cudaMalloc((void **) &d_bodies.z, bytes);
	cudaMalloc((void **) &d_bodies.vx, bytes);
	cudaMalloc((void **) &d_bodies.vy, bytes);
	cudaMalloc((void **) &d_bodies.vz, bytes);

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {
		cudaEventRecord(iter_start, 0);

		// In the first iteration both initial coordinates and velocity needs to be copied to device
		if (iter == 1) {
			cudaMemcpy(d_bodies.x,  bodies.x,  bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bodies.y,  bodies.y,  bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bodies.z,  bodies.z,  bytes, cudaMemcpyHostToDevice);

			cudaMemcpy(d_bodies.vx, bodies.vx, bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bodies.vy, bodies.vy, bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bodies.vz, bodies.vz, bytes, cudaMemcpyHostToDevice);
		}

		bodyForce<<<grid, block>>>(d_bodies, dt, tiles, nBodies);
		checkCudaError("bodyForce");
        cudaDeviceSynchronize();

		calculatePositions<<<grid, block>>>(d_bodies, dt, nBodies);
		cudaDeviceSynchronize();

		// Send final coordinates back to host
		if (iter == nIters) {
			cudaMemcpy(bodies.x, d_bodies.x, bytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(bodies.y, d_bodies.y, bytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(bodies.z, d_bodies.z, bytes, cudaMemcpyDeviceToHost);
		}

		/****************************** Save Final Coordinates ******************************/
		#ifdef SAVE_FINAL_COORDINATES
		if (iter == 2) {
			// Copy coordinates back to host for checking
			cudaMemcpy(bodies.x, d_bodies.x, bytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(bodies.y, d_bodies.y, bytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(bodies.z, d_bodies.z, bytes, cudaMemcpyDeviceToHost);

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

  	printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
	printf("Total time: %.3f\n", totalTime);

	/****************************** Cleanup ******************************/
	// Device
	cudaFree(d_bodies.x);
	cudaFree(d_bodies.y);
	cudaFree(d_bodies.z);
	cudaFree(d_bodies.vx);
	cudaFree(d_bodies.vy);
	cudaFree(d_bodies.vz);	
	
	// Host
	free(bodies.x);
	free(bodies.y);
	free(bodies.z);
	free(bodies.vx);
	free(bodies.vy);
	free(bodies.vz);	

	cudaDeviceReset();
}