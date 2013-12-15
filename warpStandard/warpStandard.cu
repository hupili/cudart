#include "warpStandard.cuh"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

extern __shared__ unsigned rngShmem[];

__global__ void CalcPiKernel(unsigned *state, unsigned N, unsigned *hits)
{
  unsigned rngRegs[WarpStandard_REG_COUNT];
  WarpStandard_LoadState(state, rngRegs, rngShmem);
  unsigned acc=0;
  for(unsigned i=0;i<N;i++)
  {
    unsigned long x=WarpStandard_Generate(rngRegs, rngShmem);
    unsigned long y=WarpStandard_Generate(rngRegs, rngShmem);
    x=(x*x)>>3;
    y=(y*y)>>3;
    if(x+y <= (1UL<<61))
    {
      acc++;
    }
  }
  hits[blockIdx.x*blockDim.x+threadIdx.x]=acc;
  WarpStandard_SaveState(rngRegs, rngShmem, state);
}


int main(int,char *[])
{
  int devId=-1;
  cudaDeviceProp devProps;
  cudaGetDevice(&devId);
  cudaGetDeviceProperties(&devProps, devId);
  unsigned gridSize=devProps.multiProcessorCount;
  unsigned blockSize=256;
  unsigned totalThreads=blockSize*gridSize;
  unsigned totalRngs=totalThreads/WarpStandard_K;
  unsigned rngsPerBlock=blockSize/WarpStandard_K;
  unsigned sharedMemBytesPerBlock=rngsPerBlock*WarpStandard_K*4;
  fprintf(stderr, "gridSize=%u, blockSize=%u, totalThreads=%u\n", gridSize, blockSize, totalThreads);
  unsigned seedBytes=totalRngs*4*WarpStandard_STATE_WORDS;
  std::vector<uint32_t> seedHost(seedBytes/4);
  void *seedDevice=0;
  if(cudaMalloc(&seedDevice, seedBytes))
  {
    fprintf(stderr, "Error couldn't allocate state array of size %u\n", seedBytes);
    exit(1);
  }
  int fr=open("/dev/urandom", O_RDONLY);
  if(seedBytes!=read(fr, &seedHost[0], seedBytes))
  {
    fprintf(stderr, "Couldn't seed RNGs.\n");
    exit(1);
  }
  //cudaMemcpy(seedDevice, &seedHost[0], seedBytes, cudaMemcpyHostToDevice);
  std::vector<uint32_t>hitsHost(totalThreads, 0);
  void *hitsDevice=0;
  if(cudaMalloc(&hitsDevice, totalThreads*4))
  {
    fprintf(stderr, "Error: couldn't allocate hits array of size %u.\n", totalThreads*4);
    exit(1);
  }
  if(cudaMemcpy(hitsDevice, &hitsHost[0], totalThreads*4, cudaMemcpyHostToDevice))
  {
    fprintf(stderr, "Error: couldn't copy hits array to device.\n");
    exit(1);
  }
  unsigned K=8;
  unsigned N=65536;
  double totalHits=0, totalSamples=0;
  for(unsigned i=0;i<K;i++)
  {
	N=N*2;
	double outputsPerKernel=totalThreads*double(N);
	CalcPiKernel<<<gridSize,blockSize,sharedMemBytesPerBlock>>>((unsigned*)seedDevice, N, (unsigned*)hitsDevice);
    cudaMemcpy(&hitsHost[0], hitsDevice, 4*totalThreads, cudaMemcpyDeviceToHost);
    //for(unsigned i=0;i<hitsHost.size();i++)
    //{
    //  fprintf(stdout, "hitsHost[%u]=%u\n", i, hitsHost[i]);
    //}
    totalSamples+=outputsPerKernel;
    totalHits += std::accumulate(hitsHost.begin(), hitsHost.end(), 0.0);
    double estimate=4*totalHits/totalSamples;
    fprintf(stdout, "totalHits=%lg, totalSamples=%lg\n", totalHits, totalSamples);
    fprintf(stdout, "samples=2^%lg, estimate=%.16lf, error=%lg\n", log(totalSamples)/log(2), estimate, std::abs(estimate-M_PI));
  }
  return 0;
}
