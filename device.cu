#include "parameters.h"
#include <curand_kernel.h> //cuda random generator

__global__ void setup(curandState *states, int seed_)
{
	int id=(blockIdx.x*blockDim.x+threadIdx.x);
	int seed = id*seed_;
	curand_init(seed,id,0,&states[id]);
};

__global__ void cuda_kernel(float *A, int ppt, curandState *states)
{	
	// int thread_num=blockDim.x*gridDim.x;
	int id=(blockIdx.x*blockDim.x+threadIdx.x);
	// int seed = id;
	// curand_init(seed,id,0,&states[id]);
	for(int i=0;i<ppt;i++){
		float x=curand_uniform(&states[id]); // range excludes 0.0 but includes 1.0
		float y=curand_uniform(&states[id]);
		// float x=0.5;
		// float y=0.5;
		float temp = (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5);
		if(temp<=0.25) A[id]++;
	}
};

float GPU_kernel(float *A, int ppt, int grid_num, int thread_num, int seed){
	const int SIZE = grid_num*thread_num;
	float *dA;
	curandState *dev_random;

	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	
	// Allocate Memory Space on Device
	cudaMalloc((void**)&dA,sizeof(float)*SIZE);
	cudaMalloc((void**)&dev_random,SIZE*sizeof(curandState));

	// Copy Data to be Calculated
	cudaMemcpy(dA, A, sizeof(float)*SIZE, cudaMemcpyHostToDevice);

	// Lunch SETUP Kernel
	dim3 dimGrid(grid_num);
	dim3 dimBlock(thread_num);
	setup<<<dimGrid,dimBlock>>>(dev_random,seed);

	// Start Timer
	cudaEventRecord(start, 0);

	// Lunch Kernel
	cuda_kernel<<<dimGrid,dimBlock>>>(dA,ppt,dev_random);

	// Stop Timer
	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop);

	// Copy Output back
	cudaMemcpy(A, dA, sizeof(float)*SIZE, cudaMemcpyDeviceToHost);

	// Release Memory Space on Device
	cudaFree(dA);

	// Calculate Elapsed Time
  	float usetime; 
  	cudaEventElapsedTime(&usetime, start, stop);  

	return usetime;
}

