#include <stdio.h>
#include <assert.h>

// __constant__: another address space.
// CUDA memory:
//    * constant memory
//    * global memory
// When there are data independent from threadID, 
// one can utilize constant memory
__constant__ int inc;
__device__ int sum;

__global__ void atomicAdd()
{
	// the two "atomicAdd"?
	// This is the overload of above function.
	// better to change a name
	int s = atomicAdd(&sum, inc);
	// as long as there is one thread that does not pass assertion,
	// the whole grid will terminate.
	assert((s - 1) % inc == 0);
	if (threadIdx.x == 0)
	{
		// will buffer and copy back to CPU later
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
	atomicAdd<<<3, 2>>>();

	// Copy sum from device memory to host memory synchronously.
	cudaMemcpyFromSymbol(&h_sum, sum, sizeof(int));

	// Print the result.
	printf("sum = %d\n", h_sum);

	// Cleanup.
	cudaDeviceReset();
}
