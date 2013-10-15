#include <stdio.h>
#include <mpi.h>

__global__ void square(float *d)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	d[i] = d[i] * d[i];
}

int main(int argc, char *argv[])
{
	// Initialize MPI.
	MPI_Init(&argc, &argv);

	// Get the node count and node rank.
	int commSize, commRank;
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

	// Initialize constants.
	int numThreadsPerBlock = 256;
	int numBlocksPerGrid = 10000;
	int dataSizePerNode = numThreadsPerBlock * numBlocksPerGrid;
	int dataSize = dataSizePerNode * commSize;

	// Generate some random numbers on the root node.
	float *data;
	if (commRank == 0)
	{
		data = new float[dataSize];
		for (int i = 0; i < dataSize; ++i)
		{
			data[i] = rand() / (float)RAND_MAX;
		}
	}

	// Allocate a buffer on each node.
	float *dataPerNode = new float[dataSizePerNode];

	// Dispatch a portion of the input data to each node.
	MPI_Scatter(data, dataSizePerNode, MPI_FLOAT, dataPerNode, dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Compute the square of each element on device.
	float *d;
	cudaMalloc((void **)&d, sizeof(float) * dataSizePerNode);
	cudaMemcpy(d, dataPerNode, sizeof(float) * dataSizePerNode, cudaMemcpyHostToDevice);
	square<<<numBlocksPerGrid, numThreadsPerBlock>>>(d);
	cudaMemcpy(dataPerNode, d, sizeof(float) * dataSizePerNode, cudaMemcpyDeviceToHost);
	cudaFree(d);

	// Compute the sum of
	float sumNode = 0.f;
	for (int i = 0; i < dataSizePerNode; ++i)
	{
		sumNode += dataPerNode[i];
	}


	float sumRoot;
	MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (commRank == 0)
	{
		printf("sumRoot = %f\n", sumRoot);
		float sumNode = 0.f;
		for (int i = 0; i < dataSize; ++i)
		{
			sumNode += data[i] * data[i];
		}
		printf("sumNode = %f\n", sumNode);
		delete[] data;
	}

	// Cleanup.
	delete[] dataPerNode;
	MPI_Finalize();
}
