#include <stdio.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	float *h_a = (float *)malloc(size);
	float *h_b = (float *)malloc(size);
	float *h_c = (float *)malloc(size);
	for (int i = 0; i < numElements; ++i)
	{
		h_a[i] = rand()/(float)RAND_MAX;
		h_b[i] = rand()/(float)RAND_MAX;
	}
	float *d_a;
	float *d_b;
	float *d_c;
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
		}
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaDeviceReset();
	free(h_a);
	free(h_b);
	free(h_c);
}
