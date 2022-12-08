#define EIGEN_NO_CUDA
#include "Eigen/Dense"
#include "utils/cuda_utils.hpp"
#include "utils/get_random.hpp"
#include "utils/math_utils.hpp"
#include "utils/Timer.hpp"

#include <cassert>

constexpr uint32_t N{2048};
constexpr uint32_t DATA_SIZE{sizeof(float) * N * N};
constexpr uint32_t BLOCK_SIZE{16};
constexpr uint32_t GRID_SIZE{N / BLOCK_SIZE};

__managed__ float ma[N * N];
__managed__ float mb[N * N];
__managed__ float mc[N * N];

__global__ void kernel()
{
  uint32_t const row{blockIdx.y * blockDim.y + threadIdx.y};
  uint32_t const col{blockIdx.x * blockDim.x + threadIdx.x};

  uint32_t offset{N * row};

  if ((row < N) && (col < N))
  {
    float result{0.0f};

    for (uint32_t s{0}; s < N; ++s)
    {
      result += ma[offset + s] * mb[col + s * N];
    }

    mc[offset + col] = result;
  }
}

int main(int argc, char *[])
{
  Eigen::MatrixXf a{N, N};
  Eigen::MatrixXf b{N, N};

  int deviceId{};
  gpuErrCheck(cudaGetDevice(&deviceId));

  gpuErrCheck(cudaMemPrefetchAsync(ma, DATA_SIZE, cudaCpuDeviceId, nullptr));
  gpuErrCheck(cudaMemPrefetchAsync(mb, DATA_SIZE, cudaCpuDeviceId, nullptr));

  for (uint32_t row{0}; row < N; ++row)
  {
    for (uint32_t col{0}; col < N; ++col)
    {
      a(row, col) = ma[row * N + col] = get_random();
      b(row, col) = mb[row * N + col] = get_random();
    }
  }

  Eigen::MatrixXf c{a * b};

  gpuErrCheck(cudaMemPrefetchAsync(ma, DATA_SIZE, deviceId, nullptr));
  gpuErrCheck(cudaMemPrefetchAsync(mb, DATA_SIZE, deviceId, nullptr));

  for (uint32_t i{0}; i < 10; ++i)
  {
    Timer timer{};
    kernel<<<dim3{GRID_SIZE, GRID_SIZE, 1}, dim3{BLOCK_SIZE, BLOCK_SIZE, 1}>>>();
    gpuErrCheck(cudaPeekAtLastError());

    gpuErrCheck(cudaDeviceSynchronize());
  }

  gpuErrCheck(cudaMemPrefetchAsync(mc, DATA_SIZE, cudaCpuDeviceId, nullptr));

  for (uint32_t row{0}; row < N; ++row)
  {
    for (uint32_t col{0}; col < N; ++col)
    {
      assert(fuzzy_compare(mc[N * row + col], c(row, col)));
    }
  }

  return 0;
}
