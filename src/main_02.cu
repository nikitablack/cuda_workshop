#include "utils/cuda_utils.hpp"

#include <cstdint>

__global__ void kernel()
{
  uint32_t threadId = blockDim.x * blockIdx.x + threadIdx.x;

  printf("blockId.x: %u, threadId.x (in block): %u, threadId: %u\n", blockIdx.x, threadIdx.x, threadId);

  // uint32_t blockId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
  // uint32_t threadId = (blockDim.x * blockDim.y * blockDim.z) * blockId + ((blockDim.x * blockDim.y) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
  //
  // printf("blockId: %u, threadId: %u\n", blockId, threadId);
}

int main(int argc, char *[])
{
  kernel<<<1, 1>>>();
  // kernel<<<dim3{1, 1, 1}, dim3{1, 1, 1}>>>();
  // kernel<<<dim3{1, 1, 1}, dim3{1, 1, 1}, 0, 0>>>();
  // kernel<<<2, 2>>>();
  gpuErrCheck(cudaPeekAtLastError());

  gpuErrCheck(cudaDeviceSynchronize());

  return 0;
}
