#include <stdio.h>
#include <iostream>
#include "parameters.h"
#include <numeric>   // accumulate
#include <ctime>
#include "pthread.h" // pthread mutex
using namespace std;

extern float GPU_kernel(float *A, int ppt, int grid_num, int thread_num, int seed); // using funtion declared in another code

bool in_range(double x,double y){
	double dis_sq = (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5); // distance to center square
	if(dis_sq<=0.25) return true; 	// in circle
	else return false;				// not in circle
}

long long CPU_calculate(int total_point){
	long long count=0;
	for(int i=0; i<total_point; i++){
		double x = (double) rand()/(RAND_MAX);
		double y = (double) rand()/(RAND_MAX);
		if(in_range(x,y)) count++;
	}
	return count;
}

// lock and unlock global variable
long long global_count=0;
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
void *countseries(void* data){
	int *nums = (int*) data;
	for(int i=0; i<*nums; i++){
		double x = (double) rand()/(RAND_MAX);
		double y = (double) rand()/(RAND_MAX);
		if(in_range(x,y)){
			pthread_mutex_lock(&mutex1);			
			global_count++;
			pthread_mutex_unlock(&mutex1);
		}
	}
	
}

void CPU_SINGLE_CORE(){
	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	// Random SEED
	srand(time(NULL*SEED1));

	// Start Timer
	cudaEventRecord(start, 0);

	// RUN CPU
	long long counts = CPU_calculate(TOTAL_POINTS);
	double pi_CPU = 4*counts/(double)TOTAL_POINTS;

	// Stop Timer
	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop);
	float usetime1; 
  	cudaEventElapsedTime(&usetime1, start, stop);

	// RESULT
	cout<<endl<<"CPU"<<endl;
	cout<<"get "<<counts<<", pi= "<<pi_CPU<<", time= "<<usetime1<<endl;
}

void CPU_MULTI_CORE(){
	// Random Seed
	srand(time(NULL)*SEED2);

	// PTHREAD
	pthread_t thread_array[CORE];
	int pp = TOTAL_POINTS/CORE;

	// Creat Timing Event
  	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	// Start Timer
	cudaEventRecord(start, 0);

	// PTHREAD Create
	for(int i=0; i<CORE; i++){
		pthread_create(&(thread_array[i]), NULL, countseries, (void*)(&pp));
	}
	for(int i=0; i<CORE; i++){
		pthread_join(thread_array[i], NULL);
	}
	double pi_CPU_multi = 4*global_count/(double)TOTAL_POINTS;

	// Stop Timer
	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop);
	float usetime2; 
  	cudaEventElapsedTime(&usetime2, start, stop);
	
	// Result
	cout<<endl<<"CPU MULTICORE"<<endl;
	cout<<"get "<<global_count<<", pi= "<<pi_CPU_multi<<", time= "<<usetime2<<endl;
}

int main()
{	
	cout<<endl<<"Simulation 1000000 times"<<endl;
	// CPU SINGLE CORE
	CPU_SINGLE_CORE();	
	
	// CPU MULTICORE
	CPU_MULTI_CORE();

	// GPU MULTI THREAD
	for(int test=0; test<NUM; test++){
		srand(time(NULL)*SEED3);
		int seed = rand();
		int grid_num = GRID[test];
		int thread_num = THREAD[test];
		int TOTAL_THREAD = grid_num*thread_num;
		int POINTS_per_THREAD = TOTAL_POINTS/TOTAL_THREAD;
		// declare a dynamic array
		// array_ptr is a pointer pointing to the first element of array
		float *array_ptr=new float[TOTAL_THREAD]; // adjacency matrix
	
		for(int i=0;i<TOTAL_THREAD;i++){
			array_ptr[i]=0;
		}
		float usetime3 = GPU_kernel(array_ptr, POINTS_per_THREAD, grid_num, thread_num, seed);
		long long sum=0;
		sum=accumulate(array_ptr,array_ptr+TOTAL_THREAD,sum);
		double pi_GPU=4*sum/(double)TOTAL_POINTS;
		// RESULT
		cout<<endl<<"GPU "<<grid_num<<"x"<<thread_num<<endl;
		cout<<"get "<<sum<<", pi= "<<pi_GPU<<", time= "<<usetime3/1000<<endl;
	}
	
	cout<<endl<<endl<<endl;
	// Please press any key to exit the program
	getchar();
}
