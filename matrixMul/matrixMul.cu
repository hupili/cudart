#include <stdio.h>

template <int BLOCK_SIZE> __global__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
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

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

int main(int argc, char **argv)
{
	const int block_size = 32;
	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)malloc(mem_size_B);
	for (int i = 0; i < size_A; ++i)
	{
		h_A[i] = 1.0f;
	}
	const float valB = 0.01f;
	for (int i = 0; i < size_B; ++i)
	{
		h_B[i] = valB;
	}
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = (float *)malloc(mem_size_C);
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, mem_size_A);
	cudaMalloc((void **)&d_B, mem_size_B);
	cudaMalloc((void **)&d_C, mem_size_C);
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
	matrixMulCUDA<block_size><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	cudaDeviceSynchronize();

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	int nIter = 300;
	for (int j = 0; j < nIter; j++)
	{
		matrixMulCUDA<block_size><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float msecTotal;
	cudaEventElapsedTime(&msecTotal, start, stop);

	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	double eps = 1.e-6 ;
	for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
	{
		double abs_err = fabs(h_C[i] - (dimsA.x * valB));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err/abs_val/dot_length ;
		if (rel_err > eps)
		{
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
		}
	}
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaDeviceReset();
}
