#define EIGEN_NO_CUDA
#include "Eigen/Dense"
#include "utils/cuda_utils.hpp"
#include "utils/get_random.hpp"
#include "utils/math_utils.hpp"
#include "utils/Timer.hpp"

#include <cassert>

constexpr uint32_t N{2048};
constexpr uint32_t DATA_SIZE{sizeof(float) * N * N};

__managed__ float ma[N * N];
__managed__ float mb[N * N];
__managed__ float mc[N * N];

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

  cublasHandle_t handle;
  cublasErrCheck(cublasCreate(&handle));

  float alpha{1.0f};
  float beta{0.0f};

  for (uint32_t i{0}; i < 10; ++i)
  {
    Timer timer{};

    cublasErrCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, ma, N, mb, N, &beta, mc, N));
    gpuErrCheck(cudaDeviceSynchronize());
  }

  cublasErrCheck(cublasDestroy(handle));

  gpuErrCheck(cudaMemPrefetchAsync(mc, DATA_SIZE, cudaCpuDeviceId, nullptr));

  for (uint32_t row{0}; row < N; ++row)
  {
    for (uint32_t col{0}; col < N; ++col)
    {
      // if (mc[N * row + col] != c(row, col))
      // {
      //   std::cout << row << " " << col << " " << mc[N * row + col] << " " << c(row, col) << '\n';
      // }
      assert(fuzzy_compare(mc[N * row + col], c(col, row)));
    }
  }

  return 0;
}
