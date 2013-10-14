#include <stdio.h>
#include <sys/time.h>

double shrDeltaT()
{
	static struct timeval old_time;
	struct timeval new_time;
	gettimeofday(&new_time, NULL);
	const double DeltaT = ((double)new_time.tv_sec + 1.0e-6 * (double)new_time.tv_usec) - ((double)old_time.tv_sec + 1.0e-6 * (double)old_time.tv_usec);
	old_time.tv_sec  = new_time.tv_sec;
	old_time.tv_usec = new_time.tv_usec;
	return DeltaT;
}

int main(int argc, char* argv[])
{
	// Initialize 4 transfer sizes, i.e. 3KB, 15KB, 15MB and 100MB.
	const int n = 4;
	const size_t sizes[n] = { 3 << 10, 15 << 10, 15 << 20, 100 << 20 };

	// Initialize the number of transfer iterations, i.e. 60K, 60K, 300 and 30 iterations, respectively.
	const int iterations[n] = { 60000, 60000, 300, 30 };

	// Print header in CSV format.
	printf("size (B),memory,direction,bandwidth (MB/s)\n");

	// Loop through the 4 transfer sizes.
	for (int s = 0; s < n; ++s)
	{
		// Calculate the total transfer size.
		const size_t size = sizes[s];
		const int iteration = iterations[s];
		const double totalSizeInMB = (double)size * iteration / (1 << 20);
		double time;

		// Allocate d_p in device memory.
		void* h_p;
		void* d_p;
		cudaMalloc(&d_p, size);

		// Allocate pageable h_p in host memory.
		h_p = malloc(size);

		// Test host-to-device bandwidth.
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
		}
		time = shrDeltaT();
		printf("%lu,%s,%s,%.0f\n", size, "pageable", "HtoD", totalSizeInMB / time);

		// Test device-to-host bandwidth.
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);
		}
		time = shrDeltaT();
		printf("%lu,%s,%s,%.0f\n", size, "pageable", "DtoH", totalSizeInMB / time);

		// Deallocate pageable h_p in host memory.
		free(h_p);

		// Allocate pinned h_p in host memory.
        cudaHostAlloc(&h_p, size, cudaHostAllocDefault);

		// Test host-to-device bandwidth.
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpyAsync(d_p, h_p, size, cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();
		time = shrDeltaT();
		printf("%lu,%s,%s,%.0f\n", size, "pinned", "HtoD", totalSizeInMB / time);

		// Test device-to-host bandwidth.
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpyAsync(h_p, d_p, size, cudaMemcpyDeviceToHost);
		}
		cudaDeviceSynchronize();
		time = shrDeltaT();
		printf("%lu,%s,%s,%.0f\n", size, "pinned", "DtoH", totalSizeInMB / time);
		// Deallocate pinned h_p in host memory.
		cudaFreeHost(h_p);

		// Deallocate d_p in device memory.
		cudaFree(d_p);
	}

	// Cleanup.
	cudaDeviceReset();
}
