#include <stdio.h>

// the following is called a "kernel"
//void vectorAdd(const float *a, const float *b, float *c, int numElements)
__global__ void vectorAdd(const float *a, const float *b, float *c, int numElements)
{
	// blockDim/ threadIdx is built-in vars, 3-dimensional (get .y, .z?)
	// blockDim: number of threads in a block
	// blockIdx: the block index
	// threadIdx: the index of the thread within the block
	// i: the global thread ID
	// more about the sizes: http://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		// Even if the condition evaluates to false, 
		// this block will still be executed,
		// just the result is not written back.
		// When come to a branch, the thread will get a mask.
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char *argv[])
{
	int numElements = 5e+4;

	// Allocate vectors a, b and c in host memory.
	size_t numBytes = sizeof(float) * numElements;
	float *h_a = (float *)malloc(numBytes);
	float *h_b = (float *)malloc(numBytes);
	float *h_c = (float *)malloc(numBytes);

	// Initialize vectors a and b.
	for (int i = 0; i < numElements; ++i)
	{
		h_a[i] = rand() / (float)RAND_MAX;
		h_b[i] = rand() / (float)RAND_MAX;
	}

	// Allocate vectors a, b and c in device memory.
	float *d_a;
	float *d_b;
	float *d_c;
	cudaMalloc((void **)&d_a, numBytes);
	cudaMalloc((void **)&d_b, numBytes);
	cudaMalloc((void **)&d_c, numBytes);

	// Copy vectors a and b from host memory to device memory synchronously.
	// synchronized operation
	cudaMemcpy(d_a, h_a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, numBytes, cudaMemcpyHostToDevice);

	// Determine the number of threads per block and the number of blocks per grid.
	int numThreadsPerBlock = 256;
	int numBlocksPerGrid = (numElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

	// Invoke the kernel on device asynchronously.
	// The following return immediately
	vectorAdd<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a, d_b, d_c, numElements);

	// Copy vector c from device memory to host memory synchronously.
	// The following is synchronous.
	// It will not execute until previous vectorAdd is finished.
	cudaMemcpy(h_c, d_c, numBytes, cudaMemcpyDeviceToHost);

	// Validate the result.
	for (int i = 0; i < numElements; ++i)
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
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	cudaDeviceReset();
	free(h_c);
	free(h_b);
	free(h_a);
}
