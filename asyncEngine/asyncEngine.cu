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
	int numElements = 5e+4;

	// Allocate vectors a, b and c in host memory.
	size_t numBytes = sizeof(float) * numElements;
	float *h_a;
	float *h_b;
	float *h_c;
	cudaMallocHost((void **)&h_a, numBytes);
	cudaMallocHost((void **)&h_b, numBytes);
	cudaMallocHost((void **)&h_c, numBytes);

	// Initialize vectors a and b.
	for (int i = 0; i < numElements; ++i)
	{
		h_a[i] = rand() / (float)RAND_MAX;
		h_b[i] = rand() / (float)RAND_MAX;
	}

	// Initialize a number of CUDA streams.
	int numStreams = 8;
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * numStreams);
    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

	// Compute the average number of elements per device and the number of spare elements.
	int avgElementsPerStream = numElements / numStreams;
	int sprElements = numElements - avgElementsPerStream * numStreams;

	float **d_a = (float **)malloc(sizeof(float *) * numStreams);
	float **d_b = (float **)malloc(sizeof(float *) * numStreams);
	float **d_c = (float **)malloc(sizeof(float *) * numStreams);

	for (int i = 0, offset = 0; i < numStreams; ++i)
	{
		// Determine the number of elements to be processed by the current device.
		int numElementsCurrentStream = avgElementsPerStream + (i < sprElements);

		// Allocate vectors a, b and c in device memory.
		size_t numBytesCurrentStream = sizeof(int) * numElementsCurrentStream;
		cudaMalloc((void **)&d_a[i], numBytesCurrentStream);
		cudaMalloc((void **)&d_b[i], numBytesCurrentStream);
		cudaMalloc((void **)&d_c[i], numBytesCurrentStream);

		// Copy vectors a and b from host memory to device memory asynchronously.
		cudaMemcpyAsync(d_a[i], h_a + offset, numBytesCurrentStream, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_b[i], h_b + offset, numBytesCurrentStream, cudaMemcpyHostToDevice, streams[i]);

		// Determine the number of threads per block and the number of blocks per grid.
		unsigned int numThreadsPerBlock = 256;
		unsigned int numBlocksPerGrid = (numElementsCurrentStream + numThreadsPerBlock - 1) / numThreadsPerBlock;

		// Invoke the kernel on device asynchronously.
		vectorAdd<<<numBlocksPerGrid, numThreadsPerBlock, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], numElementsCurrentStream);

		// Copy vector c from device memory to host memory asynchronously.
		cudaMemcpyAsync(h_c + offset, d_c[i], numBytesCurrentStream, cudaMemcpyDeviceToHost, streams[i]);

		// Increase offset to point to the next portion of data.
		offset += numElementsCurrentStream;
	}

	// Wait for the device to finish.
	cudaDeviceSynchronize();

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
	for (int i = 0; i < numStreams; ++i)
	{
		cudaFree(d_c[i]);
		cudaFree(d_b[i]);
		cudaFree(d_a[i]);
	}
	free(d_c);
	free(d_b);
	free(d_a);
    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
	cudaFreeHost(h_c);
	cudaFreeHost(h_b);
	cudaFreeHost(h_a);
	cudaDeviceReset();
}
