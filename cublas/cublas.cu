#include <stdio.h>
#include <helper_functions.h>
#include <cublas_v2.h>

typedef struct _matrixSize
{
	unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	for (int i = 0; i < hA; ++i)
	{
		for (int j = 0; j < wB; ++j)
		{
			float sum = 0;
			for (int k = 0; k < wA; ++k)
			{
				sum += A[i * wA + k] * B[k * wB + j];
			}
			C[i * wB + j] = sum;
		}
	}
}

int main(int argc, char *argv[])
{
	int block_size = 32;
	sMatrixSize matrix_size;
	matrix_size.uiWA = 2 * block_size * 5;
	matrix_size.uiHA = 4 * block_size * 5;
	matrix_size.uiWB = 2 * block_size * 5;
	matrix_size.uiHB = 4 * block_size * 5;
	matrix_size.uiWC = 2 * block_size * 5;
	matrix_size.uiHC = 4 * block_size * 5;
	unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
	unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
	unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int mem_size_B = sizeof(float) * size_B;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float *h_A = (float *)malloc(mem_size_A);
	float *h_B = (float *)malloc(mem_size_B);
	float *h_C = (float *)malloc(mem_size_C);

	// Initialize h_A and h_B.
	srand(2006);
	for (int i = 0; i < size_A; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
	}
	for (int i = 0; i < size_B; ++i)
	{
		h_B[i] = rand() / (float)RAND_MAX;
	}

	// Allocate device memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, mem_size_A);
	cudaMalloc((void **)&d_B, mem_size_B);
	cudaMalloc((void **)&d_C, mem_size_C);

	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	// setup execution parameters
	dim3 numThreadsPerBlock(block_size, block_size);
	dim3 numBlocksPerGrid(matrix_size.uiWC / numThreadsPerBlock.x, matrix_size.uiHC / numThreadsPerBlock.y);

	// Initialize a cublas handle.
	cublasHandle_t handle;
	cublasCreate(&handle);

	// cublas is column primary. Need to transpose the order.
	const float alpha = 1.0f;
	const float beta  = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);

	// Measure the performance of cublasSgemm over a number of iterations.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int numIterations = 30;
	for (int j = 0; j < numIterations; j++)
	{
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);

	// Compute and print the GLOPS/s performance metric.
	printf("%.2f GFLOP/s\n", (2.0f * matrix_size.uiWA * matrix_size.uiHA * matrix_size.uiWB * numIterations * 1e-9f) / (elapsed / 1000.0f));

	// Copy result from device to host.
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	// Destroy the cublas handle after use.
	cublasDestroy(handle);

	// Compute reference solution.
	float *reference = (float *)malloc(mem_size_C);
	matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);

	// Validate the result.
	if (!sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f))
	{
		printf("error\n");
	}

	// Cleanup.
	free(reference);
	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_A);
	cudaDeviceReset();
	free(h_C);
	free(h_B);
	free(h_A);
}
