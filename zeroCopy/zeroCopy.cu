#include <stdio.h>

__global__ void zeroCopy(const float* a, const float* b, float* c)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	c[gid] = a[gid] + b[gid];
}

int main(int argc, char *argv[])
{
	const unsigned int lws = 256;
	const unsigned int gws = 1024 * lws;

	// Allocate pinned vectors a, b and c in host memory with the cudaHostAllocMapped flag.
	float* h_a;
	float* h_b;
	float* h_c;
	cudaHostAlloc((void**)&h_a, sizeof(float) * gws, cudaHostAllocMapped);
	cudaHostAlloc((void**)&h_b, sizeof(float) * gws, cudaHostAllocMapped);
	cudaHostAlloc((void**)&h_c, sizeof(float) * gws, cudaHostAllocMapped);

	// Initialize vectors a and b.
	for (int i = 0; i < gws; ++i)
	{
		h_a[i] = rand() / (float)RAND_MAX;
		h_b[i] = rand() / (float)RAND_MAX;
	}

	// Get the mapped device points.
	float* d_a;
	float* d_b;
	float* d_c;
	cudaHostGetDevicePointer(&d_a, h_a, 0);
	cudaHostGetDevicePointer(&d_b, h_b, 0);
	cudaHostGetDevicePointer(&d_c, h_c, 0);

	// Invoke the kernel.
	zeroCopy<<<gws / lws, lws>>>(d_a, d_b, d_c);
	cudaDeviceSynchronize();

	// Validate result.
	for (int i = 0; i < gws; ++i)
	{
		const float ref = h_a[i] + h_b[i];
		if (fabs(h_c[i] - ref) > 1e-7)
		{
			printf("i = %d, ref = %f, h_c[i] = %f\n", i, ref, h_c[i]);
			break;
		}
	}

	// Cleanup.
	cudaFreeHost(h_c);
	cudaFreeHost(h_b);
	cudaFreeHost(h_a);
}
