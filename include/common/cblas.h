/*
 * Self-contained stand-in for the system CBLAS header (ATLAS / OpenBLAS).
 *
 * src/common/matrix.cpp uses exactly three CBLAS entry points -- cblas_sgemm,
 * cblas_saxpy and cblas_sscal -- to implement the *host-side* Matrix class.
 * Those host paths are used for gradient checking, CPU data marshalling and
 * the softmax-tree CPU code, so they are not performance critical.
 *
 * Providing them here (instead of depending on libatlas/libopenblas) keeps the
 * Hai Platform task image free of an extra BLAS runtime dependency: the only
 * external libraries the extension needs are the CUDA ones.
 *
 * Semantics follow the standard CBLAS reference implementation.
 */

#ifndef COMMON_CBLAS_H_
#define COMMON_CBLAS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

static inline void cblas_saxpy(int N, float alpha, const float *X, int incX,
                               float *Y, int incY) {
    int i, ix, iy;
    if (N <= 0 || alpha == 0.0f) {
        return;
    }
    if (incX == 1 && incY == 1) {
        for (i = 0; i < N; i++) {
            Y[i] += alpha * X[i];
        }
        return;
    }
    ix = (incX > 0) ? 0 : (N - 1) * (-incX);
    iy = (incY > 0) ? 0 : (N - 1) * (-incY);
    for (i = 0; i < N; i++) {
        Y[iy] += alpha * X[ix];
        ix += incX;
        iy += incY;
    }
}

static inline void cblas_sscal(int N, float alpha, float *X, int incX) {
    int i, ix;
    if (N <= 0 || incX <= 0) {
        return;
    }
    if (incX == 1) {
        for (i = 0; i < N; i++) {
            X[i] *= alpha;
        }
        return;
    }
    for (i = 0, ix = 0; i < N; i++, ix += incX) {
        X[ix] *= alpha;
    }
}

static inline void cblas_sgemm(enum CBLAS_ORDER order,
                               enum CBLAS_TRANSPOSE transA,
                               enum CBLAS_TRANSPOSE transB,
                               int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc) {
    int i, j, k;
    const int aNoTrans = (transA == CblasNoTrans);
    const int bNoTrans = (transB == CblasNoTrans);
    const int rowMajor = (order == CblasRowMajor);

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            float sum = 0.0f;
            for (k = 0; k < K; k++) {
                float a, b;
                if (rowMajor) {
                    a = aNoTrans ? A[(size_t) i * lda + k] : A[(size_t) k * lda + i];
                    b = bNoTrans ? B[(size_t) k * ldb + j] : B[(size_t) j * ldb + k];
                } else {
                    a = aNoTrans ? A[(size_t) i + (size_t) k * lda] : A[(size_t) k + (size_t) i * lda];
                    b = bNoTrans ? B[(size_t) k + (size_t) j * ldb] : B[(size_t) j + (size_t) k * ldb];
                }
                sum += a * b;
            }
            {
                size_t idx = rowMajor ? ((size_t) i * ldc + j) : ((size_t) i + (size_t) j * ldc);
                if (beta == 0.0f) {
                    C[idx] = alpha * sum;
                } else {
                    C[idx] = alpha * sum + beta * C[idx];
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif  /* COMMON_CBLAS_H_ */
