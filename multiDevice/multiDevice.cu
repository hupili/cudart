#include <stdio.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		c[i] = a[i] + b[i];
	}
	for (const clock_t threshold = clock() + 1e+4; clock() < threshold;);
}

int main(int argc, char *argv[])
{
	int numElements = 3 << 22;

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

	// Get the number of CUDA devices.
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	// Compute the average number of elements per device and the number of spare elements.
	int avgElementsPerDevice = numElements / numDevices;
	int sprElements = numElements - avgElementsPerDevice * numDevices;

	float **d_a = (float **)malloc(sizeof(float *) * numDevices);
	float **d_b = (float **)malloc(sizeof(float *) * numDevices);
	float **d_c = (float **)malloc(sizeof(float *) * numDevices);

	for (int i = 0, offset = 0; i < numDevices; ++i)
	{
		// Determine the number of elements to be processed by the current device.
		int numElementsCurrentDevice = avgElementsPerDevice + (i < sprElements);

		// Set device to be used for GPU executions.
		cudaSetDevice(i);

		// Allocate vectors a, b and c in device memory.
		size_t numBytesCurrentDevice = sizeof(int) * numElementsCurrentDevice;
		cudaMalloc((void **)&d_a[i], numBytesCurrentDevice);
		cudaMalloc((void **)&d_b[i], numBytesCurrentDevice);
		cudaMalloc((void **)&d_c[i], numBytesCurrentDevice);

		// Copy vectors a and b from host memory to device memory asynchronously.
		cudaMemcpyAsync(d_a[i], h_a + offset, numBytesCurrentDevice, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_b[i], h_b + offset, numBytesCurrentDevice, cudaMemcpyHostToDevice);

		// Determine the number of threads per block and the number of blocks per grid.
		unsigned int numThreadsPerBlock = 256;
		unsigned int numBlocksPerGrid = (numElementsCurrentDevice + numThreadsPerBlock - 1) / numThreadsPerBlock;

		// Invoke the kernel on device asynchronously.
		vectorAdd<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a[i], d_b[i], d_c[i], numElementsCurrentDevice);

		// Copy vector c from device memory to host memory asynchronously.
		cudaMemcpyAsync(h_c + offset, d_c[i], numBytesCurrentDevice, cudaMemcpyDeviceToHost);

		// Increase offset to point to the next portion of data.
		offset += numElementsCurrentDevice;
	}

	// Wait for the devices to finish.
	for (int i = 0; i < numDevices; ++i)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}

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
	for (int i = 0; i < numDevices; ++i)
	{
		cudaFree(d_c[i]);
		cudaFree(d_b[i]);
		cudaFree(d_a[i]);
	}
	free(d_c);
	free(d_b);
	free(d_a);
	cudaFreeHost(h_c);
	cudaFreeHost(h_b);
	cudaFreeHost(h_a);
	for (int i = 0; i < numDevices; ++i)
	{
		cudaSetDevice(i);
		cudaDeviceReset();
	}
}
