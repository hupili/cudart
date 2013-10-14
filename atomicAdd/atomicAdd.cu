#include <stdio.h>

__constant__ int increment;
__device__ int sum;

__global__ void atomicAdd()
{
	int s = atomicAdd(&sum, increment);
	if (threadIdx.x == 0)
	{
		printf("blockIdx.x = %d, c = %d\n", blockIdx.x, s);
	}
}

int main(int argc, char *argv[])
{
	int h_increment = 10;
	int h_sum = 1;
	cudaMemcpyToSymbol(increment, &h_increment, sizeof(int));
	cudaMemcpyToSymbol(sum, &h_sum, sizeof(int));
	atomicAdd<<<2, 2>>>();
	cudaDeviceSynchronize();
	cudaMemcpyFromSymbol(&h_sum, sum, sizeof(int));
	cudaDeviceReset();
}
