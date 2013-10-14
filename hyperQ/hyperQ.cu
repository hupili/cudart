#include <stdio.h>

__global__ void hyperQ(clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

int main(int argc, char *argv[])
{
	// Initialize constants.
	const int num_milliseconds = 10;
	const int num_kernels = 2;
	const int num_streams = 32;

	// Get the major and minor compute capability version numbers.
	int major, minor;
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
	cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

	// Get the peak clock frequency in KHz of device 0.
	int clockRate;
	cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);

	// Calculate the number of clocks within a certain period.
	clock_t num_clocks = clockRate * num_milliseconds;

	// Create streams to enqueue kernels.
	cudaStream_t *streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * num_streams);
	for (int s = 0; s < num_streams; ++s)
	{
		cudaStreamCreate(&streams[s]);
	}

	// Create events to record timing data.
	cudaEvent_t beg, end;
	cudaEventCreate(&beg);
	cudaEventCreate(&end);

	// Record an event in stream 0 before kernel invocations.
	cudaEventRecord(beg, 0);

	// Enqueue kernels to streams.
	for (int s = 0; s < num_streams; ++s)
	{
		for (int k = 0; k < num_kernels; ++k)
		{
			// The 1st argument is of type dim3 and specifies the number of blocks per grid.
			// The 2nd argument is of type dim3 and specifies the number of threads per block.
			// The 3rd argument is of type size_t and specifiesthe number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory. It is an optional argument which defaults to 0.
			// The 4th argument is of type cudaStream_t and specifies the associated stream. It is an optional argument which defaults to 0.
			hyperQ<<<1, 1, 0, streams[s]>>>(num_clocks);
		}
	}

	// Record an event in stream 0 after kernel invocations.
	cudaEventRecord(end, 0);

	// Wait for the event to complete.
	cudaEventSynchronize(end);

	// Compute the elapsed time between two events.
	float elapsed;
	cudaEventElapsedTime(&elapsed, beg, end);

	// Print the results.
	printf("%d streams, each %d kernels, each %d ms\n", num_streams, num_kernels, num_milliseconds);
	printf("       SM <= 1.3:%4d ms\n", num_milliseconds * num_kernels * num_streams);
	printf("2.0 <= SM <= 3.0:%4d ms\n", num_milliseconds * (1 + (num_kernels - 1) * num_streams));
	printf("3.5 <= SM       :%4d ms\n", num_milliseconds * num_kernels);
	printf("       SM == %d.%d:%4d ms\n", major, minor, (int)elapsed);

	// Cleanup.
	cudaEventDestroy(end);
	cudaEventDestroy(beg);
	for (int s = 0; s < num_streams; ++s)
	{
		cudaStreamDestroy(streams[s]);
	}
	free(streams);
	cudaDeviceReset();
}
