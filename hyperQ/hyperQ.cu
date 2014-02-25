#include <stdio.h>

__global__ void spin(clock_t numClocks)
{
	for (const clock_t threshold = clock() + numClocks; clock() < threshold;);
}

int main(int argc, char *argv[])
{
	// Initialize constants.
	const int numMilliseconds = 10;
	const int numKernels = 2;
	const int numStreams = 32;

	// Get the major and minor compute capability version numbers.
	int major, minor;
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
	cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

	// Get the peak clock frequency in KHz of device 0.
	int clockRate;
	cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);

	// Calculate the number of clocks within a certain period.
	clock_t numClocks = clockRate * numMilliseconds;

	// Create streams to enqueue kernels.
	cudaStream_t *streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * numStreams);
	for (int s = 0; s < numStreams; ++s)
	{
		cudaStreamCreate(&streams[s]);
	}

	// Create events to record timing data.
	// Events are like bookmarks.
	cudaEvent_t beg, end;
	cudaEventCreate(&beg);
	cudaEventCreate(&end);


	// Record an event in stream 0 before kernel invocations.
	cudaEventRecord(beg, 0);

	// Enqueue kernels to streams.
	for (int s = 0; s < numStreams; ++s)
	{
		for (int k = 0; k < numKernels; ++k)
		{
			spin<<<1, 1, 0, streams[s]>>>(numClocks);
		}
	}

	// Record an event in stream 0 after kernel invocations.
	cudaEventRecord(end, 0);

	// Wait for the event to complete.
	cudaEventSynchronize(end);

	// Compute the elapsed time between two events.
	float elapsed;
	cudaEventElapsedTime(&elapsed, beg, end);

	// Print the result.
	printf("%d streams, each %d kernels, each %d ms\n", numStreams, numKernels, numMilliseconds);
	printf("       SM <= 1.3:%4d ms\n", numMilliseconds * numKernels * numStreams);
	printf("2.0 <= SM <= 3.0:%4d ms\n", numMilliseconds * (1 + (numKernels - 1) * numStreams));
	printf("3.5 <= SM       :%4d ms\n", numMilliseconds * numKernels);
	printf("       SM == %d.%d:%4d ms\n", major, minor, (int)elapsed);

	// Cleanup.
	cudaEventDestroy(end);
	cudaEventDestroy(beg);
	for (int s = 0; s < numStreams; ++s)
	{
		cudaStreamDestroy(streams[s]);
	}
	free(streams);
	cudaDeviceReset();
}
