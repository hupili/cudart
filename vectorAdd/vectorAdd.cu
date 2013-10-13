#include <stdio.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char *argv[])
{
	int numElements = 50000;

	// Allocate vectors a, b and c in host memory.
	size_t size = sizeof(float) * numElements;
	float *h_a = (float *)malloc(size);
	float *h_b = (float *)malloc(size);
	float *h_c = (float *)malloc(size);

	// Initialize vectors a and b.
	for (int i = 0; i < numElements; ++i)
	{
		h_a[i] = rand()/(float)RAND_MAX;
		h_b[i] = rand()/(float)RAND_MAX;
	}

	// Allocate vectors a, b and c in device memory.
	float *d_a;
	float *d_b;
	float *d_c;
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Copy vectors a and b from host memory to device memory synchronously.
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	// Determine the number of threads per block and the number of blocks per grid.
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	// Invoke the kernel on device asynchronously.
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);

	// Copy vector c from device memory to host memory synchronously.
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	// Validate the result.
	for (int i = 0; i < numElements; ++i)
	{
		float actual = h_c[i];
		float expected = h_a[i] + h_b[i];
		if (fabs(actual - expected) > 1e-5)
		{
			fprintf(stderr, "h_c[%d] = %f, ref = %f\n", i, actual, expected);
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
