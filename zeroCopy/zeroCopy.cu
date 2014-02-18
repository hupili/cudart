#include <stdio.h>

__global__ void vectorAdd(const float* a, const float* b, float* c)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

int main(int argc, char *argv[])
{
	// Initialize the number of threads per block and the number of blocks per grid.
	const unsigned int numThreadsPerBlock = 256;
	const unsigned int numBlocksPerGrid = 1024;
	const unsigned int numThreadsPerGrid = numThreadsPerBlock * numBlocksPerGrid;

	// Set the flag in order to allocate pinned host memory that is accessible to the device.
	cudaSetDeviceFlags(cudaDeviceMapHost);

	// Allocate **pinned** vectors a, b and c in host memory with the cudaHostAllocMapped flag so that they can be accessed by the device.
	// In vectorAdd example, the host vectors are pageable.
	// "pinned" means they always stay in memory.
	float* h_a;
	float* h_b;
	float* h_c;
	cudaHostAlloc((void**)&h_a, sizeof(float) * numThreadsPerGrid, cudaHostAllocMapped);
	cudaHostAlloc((void**)&h_b, sizeof(float) * numThreadsPerGrid, cudaHostAllocMapped);
	cudaHostAlloc((void**)&h_c, sizeof(float) * numThreadsPerGrid, cudaHostAllocMapped);

	// Initialize vectors a and b.
	for (int i = 0; i < numThreadsPerGrid; ++i)
	{
		h_a[i] = rand() / (float)RAND_MAX;
		h_b[i] = rand() / (float)RAND_MAX;
	}

	// Get the mapped pointers for the device.
	float* d_a;
	float* d_b;
	float* d_c;
	// only pointer
	cudaHostGetDevicePointer(&d_a, h_a, 0);
	cudaHostGetDevicePointer(&d_b, h_b, 0);
	cudaHostGetDevicePointer(&d_c, h_c, 0);

	// Invoke the kernel on device asynchronously.
	// implicit copy
	// more advantage on integrated system
	// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#data-transfer-between-host-and-device
	vectorAdd<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a, d_b, d_c);

	// Wait for the device to finish.
	cudaDeviceSynchronize();

	// Validate the result.
	for (int i = 0; i < numThreadsPerGrid; ++i)
	{
		float actual = h_c[i];
		float expected = h_a[i] + h_b[i];
		if (fabs(actual - expected) > 1e-7)
		{
			printf("h_c[%d] = %f, expected = %f\n", i, actual, expected);
			break;
		}
	}

	// Cleanup.
	cudaFreeHost(h_c);
	cudaFreeHost(h_b);
	cudaFreeHost(h_a);
	cudaDeviceReset();
}
