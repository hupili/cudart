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

	// Get the number of CUDA devices.
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	// Compute the average number of elements per device and the number of spare elements.
	int avgElementsPerDevice = numElements / numDevices;
	int sprElements = numElements - avgElementsPerDevice * numDevices;

	for (int i = 0, offset = 0; i < numDevices; ++i)
	{
		// Determine the number of elements to be processed by the current device.
		int numElementsCurrentDevice = avgElementsPerDevice + (i < sprElements);

		// Set device to be used for GPU executions.
		cudaSetDevice(i);

		// Allocate vectors a, b and c in device memory.
		size_t numBytesCurrentDevice = sizeof(int) * numElementsCurrentDevice;
		float *d_a;
		float *d_b;
		float *d_c;
		cudaMalloc((void **)&d_a, numBytesCurrentDevice);
		cudaMalloc((void **)&d_b, numBytesCurrentDevice);
		cudaMalloc((void **)&d_c, numBytesCurrentDevice);

		// Copy vectors a and b from host memory to device memory asynchronously.
		cudaMemcpyAsync(d_a, h_a + offset, numBytesCurrentDevice, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_b, h_b + offset, numBytesCurrentDevice, cudaMemcpyHostToDevice);

		// Determine the number of threads per block and the number of blocks per grid.
		unsigned int numThreadsPerBlock = 256;
		unsigned int numBlocksPerGrid = (numElementsCurrentDevice + numThreadsPerBlock - 1) / numThreadsPerBlock;

		// Invoke the kernel on device asynchronously.
		vectorAdd<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a, d_b, d_c, numElementsCurrentDevice);

		// Copy vector c from device memory to host memory asynchronously.
		cudaMemcpyAsync(h_c + offset, d_c, numBytesCurrentDevice, cudaMemcpyDeviceToHost);

		// Cleanup.
//		cudaFree(d_c);
//		cudaFree(d_b);
//		cudaFree(d_a);

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
//			break;
		}
	}

	// Cleanup.
	cudaFreeHost(h_c);
	cudaFreeHost(h_b);
	cudaFreeHost(h_a);
	for (int i = 0; i < numDevices; ++i)
	{
		cudaSetDevice(i);
		cudaDeviceReset();
	}
}
