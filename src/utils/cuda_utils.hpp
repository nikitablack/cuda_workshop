#include <cublas_v2.h>

#include <cstdio>

#define gpuErrCheck(ans)                      \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define cublasErrCheck(ans)                      \
    {                                            \
        cublasAssert((ans), __FILE__, __LINE__); \
    }

inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CublasAssert: %s %d\n", file, line);
        if (abort)
            exit(code);
    }
}