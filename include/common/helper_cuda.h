/*
 * Minimal re-implementation of the CUDA samples' helper_cuda.h, providing only
 * the two helpers this 2012 code base actually uses: checkCudaErrors() and
 * getLastCudaError().
 *
 * The original header shipped with the CUDA SDK and is not part of the AlexNet
 * source release, so it is reconstructed here to keep the sources unmodified
 * in spirit (same macro names, same call syntax).
 */

#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

#include <cuda_runtime.h>
#include <cublas.h>
#include <stdio.h>
#include <stdlib.h>

inline void __checkCudaErrors(cudaError_t result, const char *const func,
                              const char *const file, const int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int) result, cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int) err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) __checkCudaErrors((val), #val, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError((msg), __FILE__, __LINE__)

#endif  /* COMMON_HELPER_CUDA_H_ */
