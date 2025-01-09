#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define THREADS_PER_BLOCK 1024

typedef struct {
	float3 *position;
	float3 *velocity;
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
        bodies->position[i] = (float3){
            2.0f * (rand() / (float)RAND_MAX) - 1.0f,  // x
            2.0f * (rand() / (float)RAND_MAX) - 1.0f,  // y
            2.0f * (rand() / (float)RAND_MAX) - 1.0f   // z
        };

        bodies->velocity[i] = (float3){
            2.0f * (rand() / (float)RAND_MAX) - 1.0f,  // vx
            2.0f * (rand() / (float)RAND_MAX) - 1.0f,  // vy
            2.0f * (rand() / (float)RAND_MAX) - 1.0f   // vz
        };
    }
}

/***************** KERNEL CODE *****************/
__global__ void bodyForce(Body p, float dt, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int tile;

	// if (tid >= n) {
	// 	return;
	// }

	float dx, dy, dz;
	float distSqr, invDist, invDist3;
	float Fx = 0.0f;
	float Fy = 0.0f;
	float Fz = 0.0f;
	float3 curr_pos;

	curr_pos.x = p.position[tid].x;
	curr_pos.y = p.position[tid].y;
	curr_pos.z = p.position[tid].z;

	for (int i = 0; i < n; i++) {
		dx = p.position[i].x - curr_pos.x;
		dy = p.position[i].y - curr_pos.y;
		dz = p.position[i].z - curr_pos.z;
		distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
		invDist = 1.0f / sqrtf(distSqr);
		invDist3 = invDist * invDist * invDist;

		Fx += dx * invDist3; 
		Fy += dy * invDist3; 
		Fz += dz * invDist3;
	}

    p.velocity[tid].x += dt*Fx;
	p.velocity[tid].y += dt*Fy;
	p.velocity[tid].z += dt*Fz;
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
	int bytes = sizeof(float3)*nBodies;
	bodies.position = (float3 *)malloc(bytes);
	bodies.velocity = (float3 *)malloc(bytes);

  	randomizeBodies(&bodies, nBodies); // Init pos / vel data

	// Set geometry
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	dim3 grid((int)(ceil((float)nBodies/THREADS_PER_BLOCK)), 1, 1);

	/****************************** Device Memory Allocation ******************************/
	cudaMalloc((void **) &d_bodies.position, bytes);
	cudaMalloc((void **) &d_bodies.velocity, bytes);

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {
		cudaEventRecord(iter_start, 0);

		// In the first iteration both initial coordinates and velocity needs to be copied to device
		if (iter == 1) {
			cudaMemcpy(d_bodies.position,  bodies.position,  bytes, cudaMemcpyHostToDevice);
			cudaMemcpy(d_bodies.velocity,  bodies.velocity,  bytes, cudaMemcpyHostToDevice);
		}

		bodyForce<<<grid, block>>>(d_bodies, dt, nBodies);
		checkCudaError("bodyForce");
        cudaDeviceSynchronize();

        // Transfer velocities calculated by the kernel back to host 
	    cudaMemcpy(bodies.velocity, d_bodies.velocity, bytes, cudaMemcpyDeviceToHost);

		// Calculate new coordinates on host 
        for (int i = 0 ; i < nBodies; i++) {
            bodies.position[i].x += bodies.velocity[i].x*dt;
            bodies.position[i].y += bodies.velocity[i].y*dt;
            bodies.position[i].z += bodies.velocity[i].z*dt;
        }

        // Tranfer new coordinates back to device for next computations
		cudaMemcpy(d_bodies.position, bodies.position, bytes, cudaMemcpyHostToDevice);

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
		fprintf(fd, "%f\n", bodies.position[i].x);
		fprintf(fd, "%f\n", bodies.position[i].y);
		fprintf(fd, "%f\n", bodies.position[i].z);
	}

	fclose(fd);

	printf("Data written successfully\n");
#endif

	/****************************** Cleanup ******************************/
	// Device
	cudaFree(d_bodies.position);
	cudaFree(d_bodies.velocity);
	
	// Host
	free(bodies.position);
	free(bodies.velocity);

	cudaDeviceReset();
}