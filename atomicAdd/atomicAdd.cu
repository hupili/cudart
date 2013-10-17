#include <stdio.h>
#include <assert.h>

__constant__ int inc;
__device__ int sum;

__global__ void atomicAdd()
{
	int s = atomicAdd(&sum, inc);
	assert((s - 1) % inc == 0);
	if (threadIdx.x == 0)
	{
		printf("blockIdx.x = %d, sum = %d\n", blockIdx.x, s);
	}
}

int main(int argc, char *argv[])
{
	// Initialize inc and sum.
	int h_inc = 3;
	int h_sum = 1;

	// Copy inc and sum from host memory to device memory synchronously.
	cudaMemcpyToSymbol(inc, &h_inc, sizeof(int));
	cudaMemcpyToSymbol(sum, &h_sum, sizeof(int));

	// Invoke the kernel on device asynchronously.
	atomicAdd<<<2, 2>>>();

	// Copy sum from device memory to host memory synchronously.
	cudaMemcpyFromSymbol(&h_sum, sum, sizeof(int));

	// Print the result.
	printf("sum = %d\n", h_sum);

	// Cleanup.
	cudaDeviceReset();
}
