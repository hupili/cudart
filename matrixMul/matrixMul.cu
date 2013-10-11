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
	float *h_A = (float *)malloc(mem_size_A);
	float *h_B = (float *)malloc(mem_size_B);
	float *h_C = (float *)malloc(mem_size_C);
	for (int i = 0; i < size_A; ++i)
	{
		h_A[i] = 1.0f;
	}
	const float valB = 0.01f;
	for (int i = 0; i < size_B; ++i)
	{
		h_B[i] = valB;
	}
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, mem_size_A);
	cudaMalloc((void **)&d_B, mem_size_B);
	cudaMalloc((void **)&d_C, mem_size_C);
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(block_size, block_size);
	dim3 blocksPerGrid(dimsB.x / threadsPerBlock.x, dimsA.y / threadsPerBlock.y);
	matrixMul<block_size><<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	cudaDeviceSynchronize();

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int nIter = 300;
	for (int i = 0; i < nIter; ++i)
	{
		matrixMul<block_size><<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float msecTotal;
	cudaEventElapsedTime(&msecTotal, start, stop);
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threadsPerBlock.x * threadsPerBlock.y);

	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	for (int i = 0; i < dimsC.x * dimsC.y; ++i)
	{
		double ref = dimsA.x * valB;
		if (fabs(h_C[i] - ref) / fabs(h_C[i]) / dimsA.x > 1.e-6)
		{
			printf("Matrix[%05d]=%.8f, ref=%.8f\n", i, h_C[i], ref);
		}
	}
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
	free(h_A);
	free(h_B);
	free(h_C);
}
