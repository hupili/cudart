#include <stdio.h>

template <int BLOCK_SIZE> __global__ void matrixMul(float *C, float *A, float *B, int wA, int wB)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd   = aBegin + wA - 1;
	int aStep  = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * bx;
	int bStep  = BLOCK_SIZE * wB;
	float Csub = 0;
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];
		__syncthreads();
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}
		__syncthreads();
	}
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

int main(int argc, char **argv)
{
	// Initialize constants.
	const int block_size = 32;
	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int size_C = dimsC.x * dimsC.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int mem_size_B = sizeof(float) * size_B;
	unsigned int mem_size_C = sizeof(float) * size_C;

	// Allocate matrices a, b and c in host memory.
	float *h_A = (float *)malloc(mem_size_A);
	float *h_B = (float *)malloc(mem_size_B);
	float *h_C = (float *)malloc(mem_size_C);

	// Initialize matrices a and b.
	for (int i = 0; i < size_A; ++i)
	{
		h_A[i] = 1.0f;
	}
	const float valB = 0.01f;
	for (int i = 0; i < size_B; ++i)
	{
		h_B[i] = valB;
	}

	// Allocate matrices a, b and c in device memory.
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, mem_size_A);
	cudaMalloc((void **)&d_B, mem_size_B);
	cudaMalloc((void **)&d_C, mem_size_C);

	// Copy matrices a and b from host memory to device memory.
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	// Determine the number of threads per block and the number of blocks per grid.
	dim3 threadsPerBlock(block_size, block_size);
	dim3 blocksPerGrid(dimsB.x / threadsPerBlock.x, dimsA.y / threadsPerBlock.y);

	// Invoke the kernel on device asynchronously.
	matrixMul<block_size><<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

	// Wait for the device to finish.
	cudaDeviceSynchronize();

	// Create events to record timing data.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record an event in stream 0 before kernel invocations.
	cudaEventRecord(start, 0);

	// Invoke the kernel for a number of iterations.
	int num_iterations = 300;
	for (int i = 0; i < num_iterations; ++i)
	{
		matrixMul<block_size><<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}

	// Record an event in stream 0 after kernel invocations.
	cudaEventRecord(stop, 0);

	// Wait for the event to complete.
	cudaEventSynchronize(stop);

	// Compute the elapsed time between two events.
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);

	// Compute and print the GLOPS/s performance metric.
	printf("%.2f GFLOP/s\n", (2.0f * dimsA.x * dimsA.y * dimsB.x * num_iterations * 1e-9f) / (elapsed / 1000.0f));

	// Copy matrix c from device memory to host memory synchronously.
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	// Validate the result.
	for (int i = 0; i < dimsC.x * dimsC.y; ++i)
	{
		float actual = h_C[i];
		float expected = dimsA.x * valB;
		if (fabs(actual - expected) / fabs(actual) / dimsA.x > 1e-7)
		{
			printf("h_C[%d]=%f, expected=%f\n", i, actual, expected);
		}
	}

	// Cleanup.
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
	free(h_A);
	free(h_B);
	free(h_C);
}
