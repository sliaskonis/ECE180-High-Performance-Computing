#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define THREADS_PER_BLOCK 1024

typedef struct { 
	float3 position;
	float3 velocity;
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
		bodies[i].position.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		bodies[i].position.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		bodies[i].position.z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		bodies[i].velocity.x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		bodies[i].velocity.y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		bodies[i].velocity.z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  	}
}

/***************** KERNEL CODE *****************/
__global__ void bodyForce(Body *p, float dt, int n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	float dx, dy, dz;
	float distSqr, invDist, invDist3;
	float Fx = 0.0f;
	float Fy = 0.0f;
	float Fz = 0.0f;

	for (int i = 0; i < n; i++) {
		dx = p[i].position.x - p[tid].position.x;
		dy = p[i].position.y - p[tid].position.y;
		dz = p[i].position.z - p[tid].position.z;
		distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
		invDist = 1.0f / sqrtf(distSqr);
		invDist3 = invDist * invDist * invDist;

		Fx += dx * invDist3; 
        Fy += dy * invDist3; 
        Fz += dz * invDist3;
	}

    p[tid].velocity.x += dt*Fx;
	p[tid].velocity.y += dt*Fy;
	p[tid].velocity.z += dt*Fz;
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

  	randomizeBodies(p, nBodies); // Init pos / vel data

	// Set geometry
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	dim3 grid((int)(ceil((float)nBodies/THREADS_PER_BLOCK)), 1, 1);

	/****************************** Device Data Allocation ******************************/
	cudaMalloc((void **) &d_buf, bytes);
	d_p = (Body*)d_buf;

	/****************************** Real Computation ******************************/
  	for (int iter = 1; iter <= nIters; iter++) {

		cudaEventRecord(iter_start, 0);

		// For the first iteration:	- Transfer initial coordinates to device
		if (iter == 1) {
			cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
		}

		bodyForce<<<grid, block>>>(d_p, dt, nBodies);
		checkCudaError("bodyForce");
        cudaDeviceSynchronize();

        // Transfer data back to host in order to compute new coordinates
	    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

        for (int i = 0 ; i < nBodies; i++) { // integrate position
            p[i].position.x += p[i].velocity.x*dt;
            p[i].position.y += p[i].velocity.y*dt;
            p[i].position.z += p[i].velocity.z*dt;
        }

        // Tranfer new coordinates back to device for next computations
	    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);

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
		fprintf(fd, "%f\n", p[i].position.x);
		fprintf(fd, "%f\n", p[i].position.y);
		fprintf(fd, "%f\n", p[i].position.z);
	}

	fclose(fd);

	printf("Data written successfully\n");
#endif
	
	/****************************** Cleanup ******************************/
	cudaFree(d_buf);
	free(buf);
	
	cudaDeviceReset();
}