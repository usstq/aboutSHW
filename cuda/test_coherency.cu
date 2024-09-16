#include "cuda_utils.h"
// https://forums.developer.nvidia.com/t/what-happens-to-the-gpu-cache-at-the-end-of-the-kernel/45298/9

/*
As a result, GPU caches are usually incoherent, 
and require explicit flushing and/or invalidation
of the caches in order to recohere
(i.e. to have a coherent view of the data between
the GPU cores and/or other devices).


Note, shared memory is not cache, and it's coherent since it's
connected to CUDA cores w/o cache.

I didn't find any cache flush
*/

#include <stdio.h>

__global__ void w(int *data, const int val, const int sz){

  for (int i = threadIdx.x+blockDim.x*blockIdx.x; i< sz; i+=gridDim.x*blockDim.x)
    data[i] = val;
}

__global__ void r(int *data, int *r, const int sz){
  int val;
  for (int i = threadIdx.x+blockDim.x*blockIdx.x; i< sz; i+=gridDim.x*blockDim.x)
    val += data[i];
  if (val == 0) *r = val;
}


int main(){

  const int s = 1024*1024;  // 1M
  const int sz = s*sizeof(int);  // 4MB
  int *d1, *d2, *res;
  cudaMalloc(&d1, sz*10);
  cudaMalloc(&d2, sz*10);
  cudaMalloc(&res, sizeof(int));
  cudaMemset(d1, 1, sz);
  cudaMemset(d2, 1, sz);
  w<<<160,1024>>>(d2, 1, s);
  r<<<160,1024>>>(d1, res, s);
  w<<<160,1024>>>(d1, 1, s);
  r<<<160,1024>>>(d1, res, s);
  cudaDeviceSynchronize();
}
