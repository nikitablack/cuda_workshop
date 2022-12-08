#include "utils/cuda_utils.hpp"
#include "utils/get_random.hpp"
#include "utils/Timer.hpp"

constexpr uint32_t N{256};
constexpr uint32_t DATA_SIZE{sizeof(float) * N * N};

__managed__ float ma[N * N];
__managed__ float mb[N * N];
__managed__ float mc[N * N];

__global__ void kernel()
{
  for (uint32_t row{0}; row < N; ++row)
  {
    uint32_t offset{N * row};

    for (uint32_t col{0}; col < N; ++col)
    {
      float result{0.0f};

      for (uint32_t s{0}; s < N; ++s)
      {
        result += ma[offset + s] * mb[col + s * N];
      }

      mc[offset + col] = result;
    }
  }
}

int main(int argc, char *[])
{
  // int deviceId{};
  // gpuErrCheck(cudaGetDevice(&deviceId));

  // gpuErrCheck(cudaMemPrefetchAsync(ma, DATA_SIZE, cudaCpuDeviceId, nullptr));
  // gpuErrCheck(cudaMemPrefetchAsync(mb, DATA_SIZE, cudaCpuDeviceId, nullptr));

  for (uint32_t i{0}; i < N * N; ++i)
  {
    ma[i] = get_random();
    mb[i] = get_random();
  }

  // gpuErrCheck(cudaMemPrefetchAsync(ma, DATA_SIZE, deviceId, nullptr));
  // gpuErrCheck(cudaMemPrefetchAsync(mb, DATA_SIZE, deviceId, nullptr));

  for (uint32_t i{0}; i < 10; ++i)
  {
    Timer timer{};

    kernel<<<1, 1>>>();
    gpuErrCheck(cudaPeekAtLastError());

    gpuErrCheck(cudaDeviceSynchronize());
  }

  // gpuErrCheck(cudaMemPrefetchAsync(mc, DATA_SIZE, cudaCpuDeviceId, nullptr));

  return 0;
}
