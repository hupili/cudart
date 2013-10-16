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

	// allocate host memory for the result
	float *h_CUBLAS = (float *)malloc(mem_size_C);

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

	// execute the kernel
	cublasHandle_t handle;
	cublasCreate(&handle);

	const float alpha = 1.0f;
	const float beta  = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int numIterations = 30;
	for (int j = 0; j < numIterations; j++)
	{
		// cublas is column primary. Need to transpose the order.
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / numIterations;
	double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiWA * (double)matrix_size.uiHA * (double)matrix_size.uiWB;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

	// copy result from device to host
	cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	cublasDestroy(handle);

	// compute reference solution
	float *reference = (float *)malloc(mem_size_C);
	matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);

	// Validate the result.
	if (!sdkCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f))
	{
		printf("error\n");
	}

	// Cleanup.
	free(h_A);
	free(h_B);
	free(h_C);
	free(reference);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
}
