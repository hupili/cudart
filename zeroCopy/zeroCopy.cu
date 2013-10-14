#include <stdio.h>

__global__ void zeroCopy(const float* a, const float* b, float* c)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	c[gid] = a[gid] + b[gid];
}

int main(int argc, char *argv[])
{
	// Initialize local work size and global work size.
	// In CUDA terms, they are the number of threads per block and the number of threads per grid, respectively.
	// gws / lws yields the number of blocks per grid.
	const unsigned int lws = 256;
	const unsigned int gws = 1024 * lws;

	// Allocate pinned vectors a, b and c in host memory with the cudaHostAllocMapped flag so that they can be accessed by the device.
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

	// Get the mapped pointers for the device.
	// On integrated systems where device memory and host memory are physically the same, the mapping mechanism saves superfluous copies from host to device.
	// The 'integrated' field of device properties indicates whether a system is integrated or not.
	float* d_a;
	float* d_b;
	float* d_c;
	cudaHostGetDevicePointer(&d_a, h_a, 0);
	cudaHostGetDevicePointer(&d_b, h_b, 0);
	cudaHostGetDevicePointer(&d_c, h_c, 0);

	// Invoke the kernel on device asynchronously.
	zeroCopy<<<gws / lws, lws>>>(d_a, d_b, d_c);

	// Wait for the device to finish.
	cudaDeviceSynchronize();

	// Validate the result.
	for (int i = 0; i < gws; ++i)
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
	cudaFreeHost(h_c);
	cudaFreeHost(h_b);
	cudaFreeHost(h_a);
	cudaDeviceReset();
}
