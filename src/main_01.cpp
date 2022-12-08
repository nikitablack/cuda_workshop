#include "utils/get_random.hpp"
#include "utils/Timer.hpp"
#include "Eigen/Dense"

constexpr uint32_t N{2048};

int main(int argc, char *[])
{
  Eigen::MatrixXf a{N, N};
  Eigen::MatrixXf b{N, N};

  for (uint32_t row{0}; row < N; ++row)
  {
    for (uint32_t col{0}; col < N; ++col)
    {
      a(row, col) = get_random();
      b(row, col) = get_random();
    }
  }

  for (uint32_t i{0}; i < 10; ++i)
  {
    Timer timer{};

    Eigen::MatrixXf c = a * b;
  }

  return 0;
}
