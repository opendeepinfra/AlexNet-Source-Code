/*
 * Minimal re-implementation of the CUDA SDK's cutil_inline.h.
 *
 * The 2012 sources include <cutil_inline.h> (shipped with the CUDA 4.0 SDK,
 * not part of the AlexNet release) and use exactly one helper from it:
 * cutilCheckMsg().  Reconstructed here with the original semantics.
 */

#ifndef COMMON_CUTIL_INLINE_H_
#define COMMON_CUTIL_INLINE_H_

#include <cuda_runtime.h>
#include <cublas.h>
#include <stdio.h>
#include <stdlib.h>

/* The CUDA 4.0 SDK header also supplied these; the 2012 kernels use them. */
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

inline void cutilCheckMsg(const char *errorMessage) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUTIL CUDA error : %s : (%d) %s.\n",
                errorMessage, (int) err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void cutilSafeCall(cudaError_t err) {
    if (cudaSuccess != err) {
        fprintf(stderr, "CUTIL CUDA error : (%d) %s.\n",
                (int) err, cudaGetErrorString(err));
        exit(-1);
    }
}

#define CUDA_SAFE_CALL_NO_SYNC(hr) cutilSafeCall(hr)
#define CUDA_SAFE_CALL(hr)         cutilSafeCall(hr)

#endif  /* COMMON_CUTIL_INLINE_H_ */
