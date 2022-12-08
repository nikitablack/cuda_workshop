#define EIGEN_NO_CUDA
#include "Eigen/Dense"
#include "utils/cuda_utils.hpp"
#include "utils/get_random.hpp"
#include "utils/math_utils.hpp"
#include "utils/Timer.hpp"

#include <cassert>

constexpr uint32_t N{512};
constexpr uint32_t DATA_SIZE{sizeof(float) * N * N};

__global__ void kernel(float const *const da, float const *const db, float *dc)
{
  for (uint32_t row{0}; row < N; ++row)
  {
    uint32_t offset{N * row};

    for (uint32_t col{0}; col < N; ++col)
    {
      float result{0.0f};

      for (uint32_t s{0}; s < N; ++s)
      {
        result += da[offset + s] * db[col + s * N];
      }

      dc[offset + col] = result;
    }
  }
}

int main(int argc, char *[])
{
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a{N, N};
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b{N, N};

  for (uint32_t row{0}; row < N; ++row)
  {
    for (uint32_t col{0}; col < N; ++col)
    {
      a(row, col) = get_random();
      b(row, col) = get_random();
    }
  }

  Eigen::MatrixXf c{a * b};

  float *da{nullptr};
  float *db{nullptr};
  float *dc{nullptr};

  gpuErrCheck(cudaMalloc(&da, DATA_SIZE));
  gpuErrCheck(cudaMalloc(&db, DATA_SIZE));
  gpuErrCheck(cudaMalloc(&dc, DATA_SIZE));

  gpuErrCheck(cudaMemcpy(da, a.data(), DATA_SIZE, cudaMemcpyHostToDevice));
  gpuErrCheck(cudaMemcpy(db, b.data(), DATA_SIZE, cudaMemcpyHostToDevice));

  {
    Timer timer{};
    kernel<<<1, 1>>>(da, db, dc);
    gpuErrCheck(cudaPeekAtLastError());

    gpuErrCheck(cudaDeviceSynchronize());
  }

  float *result{new float[N * N]};
  gpuErrCheck(cudaMemcpy(result, dc, DATA_SIZE, cudaMemcpyDeviceToHost));

  for (uint32_t row{0}; row < N; ++row)
  {
    for (uint32_t col{0}; col < N; ++col)
    {
      assert(fuzzy_compare(result[N * row + col], c(row, col)));
    }
  }

  delete[] result;

  gpuErrCheck(cudaFree(da));
  gpuErrCheck(cudaFree(db));
  gpuErrCheck(cudaFree(dc));

  return 0;
}
