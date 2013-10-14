#include <helper_cuda.h>

__global__ void iota(float *a)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i] = i;
}

int main(int argc, char *argv[])
{
	int numElements = 1e+10;
	size_t size = sizeof(float) * numElements;

	// Allocate vector a in device memory.
	float *d_a;
	checkCudaErrors(cudaMalloc((void **)&d_a, size));

	// Determine the number of threads per block and the number of blocks per grid.
	int numThreadsPerBlock = 256;
	int numBlocksPerGrid = (numElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

	// Invoke the kernel on device asynchronously.
	iota<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a);
	checkCudaErrors(cudaGetLastError());

	// Cleanup.
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaDeviceReset());
}
