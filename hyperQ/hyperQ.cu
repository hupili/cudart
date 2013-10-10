#include <stdio.h>
#include <time.h>

__global__ void hyperQ(clock_t num_clocks)
{
	for (const clock_t threshold = clock() + num_clocks; clock() < threshold;);
}

int main(int argc, char *argv[])
{
	const int num_milliseconds = 10;
	const int num_kernels = 2;
	const int num_streams = 32;
	cudaStream_t *streams;
	cudaEvent_t beg, end;
	clock_t num_clocks;
	float elapsed;
	int s, k;
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);
    cudaGetDeviceProperties(&deviceProp, cuda_device);
	num_clocks = deviceProp.clockRate * num_milliseconds;
	streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * num_streams);
	for (s = 0; s < num_streams; ++s)
	{
		cudaStreamCreate(&streams[s]);
	}
	cudaEventCreate(&beg);
	cudaEventCreate(&end);
	cudaEventRecord(beg, 0);
	for (s = 0; s < num_streams; ++s)
	{
		for (k = 0; k < num_kernels; ++k)
		{
			hyperQ<<<1, 1, 0, streams[s]>>>(num_clocks);
		}
	}
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed, beg, end);
	cudaEventDestroy(end);
	cudaEventDestroy(beg);
	for (s = 0; s < num_streams; ++s)
	{
		cudaStreamDestroy(streams[s]);
	}
	free(streams);
    cudaDeviceReset();
	printf("%d streams, each %d kernels, each %d ms\n", num_streams, num_kernels, num_milliseconds);
	printf("       SM <= 1.3:%4d ms\n", num_milliseconds * num_kernels * num_streams);
	printf("2.0 <= SM <= 3.0:%4d ms\n", num_milliseconds * (1 + (num_kernels - 1) * num_streams));
	printf("3.5 <= SM       :%4d ms\n", num_milliseconds * num_kernels);
	printf("       SM == %d.%d:%4d ms\n", deviceProp.major, deviceProp.minor, (int)elapsed);
}
