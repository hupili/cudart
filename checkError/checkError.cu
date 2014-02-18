#include <helper_cuda.h>

__global__ void iota(float *a)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i] = i;
}

int main(int argc, char *argv[])
{
	// this will not give error
	int numElements = 1e+8;
	// this gives OOM error
	// actually this gives warning even during compilation.
	int numElements = 1e+10;

	// Allocate vector a in device memory.
	float *d_a;
	checkCudaErrors(cudaMalloc((void **)&d_a, sizeof(float) * numElements));

	// Determine the number of threads per block and the number of blocks per grid.
	int numThreadsPerBlock = 256;
	int numBlocksPerGrid = (numElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

	// Invoke the kernel on device asynchronously.
	iota<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a);

	// Cleanup.
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaDeviceReset());
}
