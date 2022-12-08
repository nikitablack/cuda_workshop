cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(test LANGUAGES CXX CUDA)

add_executable(test
    src/main_01.cpp
)

set_target_properties(test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"
)

set(CUDA_FLAGS -O2)
set(CXX_FLAGS -O2)

target_compile_options(test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS};>")
target_compile_options(test PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};>")

# find_package(CUDAToolkit)
# target_link_libraries(test PRIVATE CUDA::cublas)