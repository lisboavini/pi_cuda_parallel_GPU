#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

// const defines
#define NBIN 1000000000
#define NUM_BLOCK 4
#define NUM_THREAD 16

// struct to get time
struct timeval current_time = {0,0};

int tid;
float pi = 0, time_elapsed;
uint begin, end;

// function to calculate pi
__global__ void cal_pi(double *sum, int nbin, double step, int nthreads, int nblocks) 
{
	// var declaration
	int i;
	double x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x; 
	
	// pi equation calculation
	for (i=idx; i< nbin; i+=nthreads*nblocks) {
		x = i*step;
		sum[idx] += double(4.0/(1.0+(x*x)));
	}
}

int main() 
{
	// var declaration
	double *sumDev;
	double step = 1.0/NBIN; // dx
	size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);

	// initializing file in append mode to insert experiment data
	FILE *f = fopen("/home/aac-pc/Daniel/experimental-log.txt", "a");
	
	// alooc space to acc variable
	cudaMallocManaged(&sumDev, size); 
	
	// get initial time to evaluate performance
	gettimeofday(&current_time, NULL);	
	begin = current_time.tv_sec*1000000 + current_time.tv_usec;
	
	// call function to calculate pi in threads
	cal_pi<<<NUM_BLOCK, NUM_THREAD>>>(sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK);
	// synchronize threads
	cudaDeviceSynchronize();
	
	// get final time to evaluate performance
	gettimeofday(&current_time, NULL);
	end = current_time.tv_sec*1000000 + current_time.tv_usec;
	time_elapsed = end - begin;
	
	// calculate pi final value
	for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++){
		pi += sumDev[tid];
	}
	pi *= step;
	
	// print final value in console and save data info in log file
	printf("PI = %f\n",pi);
	fprintf(f, "%d;%d;%f;%f\n", NUM_THREAD, NUM_BLOCK, pi, (time_elapsed/1000000));	

	// free cuda var
	cudaFree(sumDev);

	// close file
	fclose(f);

	return 0;
}