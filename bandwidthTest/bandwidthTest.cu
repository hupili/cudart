#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
	const int n = 4;
	const size_t sizes[n] = { 3 << 10, 15 << 10, 15 << 20, 100 << 20 };
	const int iterations[n] = { 60000, 60000, 300, 30 };
	printf("size (B),memory,direction,time (s),bandwidth (MB/s)\n");
	for (int s = 0; s < n; ++s)
	{
		const size_t size = sizes[s];
		const int iteration = iterations[s];
		const double bandwidth_unit = (double)size * iteration / (1 << 20);
		void* h_p;
		void* d_p;
		double time;
		double bandwidth;
		cudaMalloc(&d_p, size);

		// allocate pageable h_p
		h_p = malloc(size);
		// --memory=pageable --access=direct --htod
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();
		time = shrDeltaT();
		bandwidth = bandwidth_unit / time;
		printf("%lu,%s,%s,%.3f,%.0f\n", size, "pageable", "HtoD", time, bandwidth);
		// --memory=pageable --access=direct --dtoh
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);
		}
		cudaDeviceSynchronize();
		time = shrDeltaT();
		bandwidth = bandwidth_unit / time;
		printf("%lu,%s,%s,%.3f,%.0f\n", size, "pageable", "DtoH", time, bandwidth);
		// deallocate pageable h_p
		free(h_p);

		// allocate pinned h_p
        cudaHostAlloc(&h_p, size, cudaHostAllocDefault);
		// --memory=pinned --access=direct --htod
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpyAsync(d_p, h_p, size, cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();
		time = shrDeltaT();
		bandwidth = bandwidth_unit / time;
		printf("%lu,%s,%s,%.3f,%.0f\n", size, "pinned", "HtoD", time, bandwidth);
		// --memory=pinned --access=direct --dtoh
		shrDeltaT();
		for (int i = 0; i < iteration; ++i)
		{
			cudaMemcpyAsync(h_p, d_p, size, cudaMemcpyDeviceToHost);
		}
		cudaDeviceSynchronize();
		time = shrDeltaT();
		bandwidth = bandwidth_unit / time;
		printf("%lu,%s,%s,%.3f,%.0f\n", size, "pinned", "DtoH", time, bandwidth);
		// deallocate pinned h_p
		cudaFreeHost(h_p);

		cudaFree(d_p);
	}
}
