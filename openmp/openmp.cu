#include <stdio.h>
#include <omp.h>

__global__ void scalarAdd(int *data, int inc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	data[i] += inc;
}

int main(int argc, char *argv[])
{
	// Get the number of CUDA devices.
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	// Allocate and initialize data.
	size_t numElements = 8192 * numDevices;
	size_t numBytes = sizeof(int) * numElements;
	int *data = (int *)malloc(numBytes);
	memset(data, 0, numBytes);
	int inc = 7;

	// Create as many CPU threads as there are CUDA devices. Each CPU thread controls a different device, processing its portion of the data.
	omp_set_num_threads(numDevices);

	// All variables declared inside an "omp parallel" scope are local to each CPU thread.
	#pragma omp parallel
	{
		// Get the number of CPU threads and the thread number of the current CPU thread.
		// 0 <= threadNum <= numThreads - 1
		int numThreads = omp_get_num_threads();
		int threadNum = omp_get_thread_num();

		// Set device to be used for GPU executions.
		int deviceNum = threadNum % numDevices;
		cudaSetDevice(deviceNum);

		// Calculate the number of elements per CPU thread and the number of bytes per CPU thread.
		size_t numElementsPerThread = numElements / numThreads;
		size_t numBytesPerThread = sizeof(int) * numElementsPerThread;

		// Calculate the offset to the original data for the current CPU thread.
		int *h = data + numElementsPerThread * threadNum;

		// Allocate device memory to temporarily hold the portion of the data of the current CPU thread.
		int *d;
		cudaMalloc((void **)&d, numBytesPerThread);

		// Copy the portion of the data of the current CPU thread from host memory to device memory.
		cudaMemcpy(d, h, numBytesPerThread, cudaMemcpyHostToDevice);

		// Invoke the kernel for the current portion of the data.
		scalarAdd<<<numElementsPerThread / 128, 128>>>(d, inc);

		// Copy the portion of the data of the current CPU thread from device memory to host memory.
		cudaMemcpy(h, d, numBytesPerThread, cudaMemcpyDeviceToHost);

		// Deallocate the temporary device memory.
		cudaFree(d);

		// Cleanup.
		cudaDeviceReset();
	}

	for (int i = 0; i < numElements; ++i)
	{
		int actual = data[i];
		int expected = 0 + inc;
		if (actual != expected)
		{
			printf("data[%d] = %d, expected = %d\n", i, actual, expected);
			break;
		}
	}

	// Cleanup.
	free(data);
}
