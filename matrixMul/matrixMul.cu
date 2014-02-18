#include <stdio.h>
#include <time.h>

// utilize global memory --> shared memory to accelerate matrix multiplication

void naiveMultiply(float *a, float *b, float *c, int M, int N, int w)
{
	for (int row = 0; row < M; ++row)
	for (int col = 0; col < N; ++col)
	{
		float sum = 0.0f;
		for (int i = 0; i < w; ++i)
		{
			sum += a[row*w+i] * b[i*N+col];
		}
		c[row*N+col] = sum;
	}
}

// This TILE_DIM must be known at compile time
template <int TILE_DIM> __global__ void simpleMultiply(float *a, float *b, float *c, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	for (int i = 0; i < TILE_DIM; ++i)
	{
		sum += a[row*TILE_DIM+i] * b[i*N+col];
	}
	c[row*N+col] = sum;
}

template <int TILE_DIM> __global__ void coalescedMultiply(float *a, float *b, float *c, int N)
{
	__shared__ float aTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
	for (int i = 0; i < TILE_DIM; i++)
	{
		sum += aTile[threadIdx.y][i]* b[i*N+col];
	}
	c[row*N+col] = sum;
}

template <int TILE_DIM> __global__ void sharedABMultiply(float *a, float *b, float *c, int N)
{
	__shared__ float aTile[TILE_DIM][TILE_DIM];
	__shared__ float bTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
	bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
	__syncthreads();
	for (int i = 0; i < TILE_DIM; ++i)
	{
		sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
	}
	c[row*N+col] = sum;
}

template <int TILE_DIM> __global__ void sharedABUnrolledMultiply(float *a, float *b, float *c, int N)
{
	__shared__ float aTile[TILE_DIM][TILE_DIM];
	__shared__ float bTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
	bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
	__syncthreads();
// unroll: compile the loop into TILE_DIM operations
// Unrolling is possible provided that TILE_DIM is known at compile time.
#pragma unroll
	for (int i = 0; i < TILE_DIM; ++i)
	{
		sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
	}
	c[row*N+col] = sum;
}

int main(int argc, char *argv[])
{
	// Initialize constants.
	const int w = 32;
	const int M = w * 59;
	const int N = w * 53;
	size_t numElementsA = M * w;
	size_t numElementsB = w * N;
	size_t numElementsC = M * N;
	size_t numBytesA = sizeof(float) * numElementsA;
	size_t numBytesB = sizeof(float) * numElementsB;
	size_t numBytesC = sizeof(float) * numElementsC;

	// Allocate matrices a, b and c in host memory.
	float *h_a = (float *)malloc(numBytesA);
	float *h_b = (float *)malloc(numBytesB);
	float *h_c = (float *)malloc(numBytesC);
	float *h_r = (float *)malloc(numBytesC);

	// Initialize matrices a and b.
	srand(time(0));
	for (int i = 0; i < numElementsA; ++i)
	{
		h_a[i] = rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < numElementsB; ++i)
	{
		h_b[i] = rand() / (float)RAND_MAX;
	}

	// Compute a reference answer in host.
	naiveMultiply(h_a, h_b, h_r, M, N, w);

	// Allocate matrices a, b and c in device memory.
	float *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, numBytesA);
	cudaMalloc((void **)&d_b, numBytesB);
	cudaMalloc((void **)&d_c, numBytesC);

	// Copy matrices a and b from host memory to device memory.
	cudaMemcpy(d_a, h_a, numBytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, numBytesB, cudaMemcpyHostToDevice);

	// Warm up the device.
	// dim3? why not dim2?
	dim3 numThreadsPerBlock(w, w);
	dim3 numBlocksPerGrid(N / numThreadsPerBlock.x, M / numThreadsPerBlock.y);
	simpleMultiply<w><<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a, d_b, d_c, N);
	cudaDeviceSynchronize();

	// Create events to record timing data.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record an event before kernel invocations.
	cudaEventRecord(start);

	// Invoke the kernel for a number of iterations.
	int numIterations = 300;
	for (int i = 0; i < numIterations; ++i)
	{
		simpleMultiply<w><<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a, d_b, d_c, N);
	}

	// Record an event after kernel invocations.
	cudaEventRecord(stop);

	// Wait for the event to complete.
	cudaEventSynchronize(stop);

	// Compute the elapsed time between two events.
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);

	// Compute and print the GLOPS/s performance metric.
	printf("%.2f GFLOP/s\n", (2.0f * M * N * w * numIterations * 1e-9f) / (elapsed / 1000.0f));

	// Copy matrix c from device memory to host memory synchronously.
	cudaMemcpy(h_c, d_c, numBytesC, cudaMemcpyDeviceToHost);

	// Validate the result.
	for (int i = 0; i < numElementsC; ++i)
	{
		float actual = h_c[i];
		float expected = h_r[i];
		if (fabs(actual - expected) / w > 1e-6)
		{
			printf("h_c[%d] = %f, expected = %f\n", i, actual, expected);
			break;
		}
	}

	// Cleanup.
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	cudaDeviceReset();
	free(h_r);
	free(h_c);
	free(h_b);
	free(h_a);
}
