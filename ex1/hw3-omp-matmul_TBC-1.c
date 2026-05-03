/*
 * Matrix-Matrix Multiplication  C = A * B
 *
 *   A : M × N
 *   B : N × K
 *   C : M × K
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DTYPE float

/* Returns wall-clock time in seconds using CLOCK_MONOTONIC. */
static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Allocate a zero-initialised M×N matrix */
DTYPE *alloc_matrix(int rows, int cols) {
  DTYPE *m = (DTYPE *)calloc((size_t)rows * cols, sizeof(DTYPE));
  if (!m) {
    fprintf(stderr, "OOM: %d x %d\n", rows, cols);
    exit(1);
  }
  return m;
}

/* Initialize matrix with reproducible values */
void init_matrix(DTYPE *A, int rows, int cols, DTYPE offset) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      A[i * cols + j] = (DTYPE)(i + j + offset) * 0.01;
}

/* Check the result matrices are correct */
int verify(const DTYPE *ref, const DTYPE *test, int M, int K) {
  for (int i = 0; i < M * K; i++) {
    DTYPE diff = fabsf(ref[i] - test[i]);
    if (diff > 0.0001 * fabsf(ref[i])) {
      fprintf(stderr, "  MISMATCH at [%d]: ref=%.6f test=%.6f\n", i, ref[i],
              test[i]);
      return 0;
    }
  }
  return 1;
}

/* 1. naive nested loops */
void matmul_serial(const DTYPE *A, const DTYPE *B, DTYPE *C, int M, int N,
                   int K) {
  for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) {
      DTYPE sum = (DTYPE)0;
      for (int j = 0; j < N; j++)
        sum += A[i * N + j] * B[j * K + k];
      C[i * K + k] = sum;
    }
}

/* 2. OpenMP parallel version */
void matmul_omp_parallel(const DTYPE *A, const DTYPE *B, DTYPE *C, int M, int N,
                         int K) {
#pragma omp parallel for
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      DTYPE sum = (DTYPE)0;
      for (int j = 0; j < N; j++)
        sum += A[i * N + j] * B[j * K + k];
      C[i * K + k] = sum;
    }
  }
}

/* 3. OpenMP SIMD version */
void matmul_omp_simd(const DTYPE *A, const DTYPE *B, DTYPE *C, int M, int N,
                     int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      DTYPE sum = (DTYPE)0;
#pragma omp simd reduction(+ : sum)
      for (int j = 0; j < N; j++)
        sum += A[i * N + j] * B[j * K + k];
      C[i * K + k] = sum;
    }
  }
}

/* 4. OpenMP parallel + SIMD version */
void matmul_omp_hybrid(const DTYPE *A, const DTYPE *B, DTYPE *C, int M, int N,
                       int K) {
#pragma omp parallel for
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      DTYPE sum = (DTYPE)0;
#pragma omp simd reduction(+ : sum)
      for (int j = 0; j < N; j++)
        sum += A[i * N + j] * B[j * K + k];
      C[i * K + k] = sum;
    }
  }
}

/* 5. OpenMP GPU offloading version */
void matmul_omp_gpu(const DTYPE *A, const DTYPE *B, DTYPE *C, int M, int N,
                    int K) {
#pragma omp target teams distribute parallel for map(                          \
        to : A[0 : M * N], B[0 : N * K]) map(from : C[0 : M * K]) collapse(2)
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      DTYPE sum = (DTYPE)0;
      for (int j = 0; j < N; j++)
        sum += A[i * N + j] * B[j * K + k];
      C[i * K + k] = sum;
    }
  }
}

int main(int argc, char *argv[]) {
  int M = 1024, N = 1024, K = 4096;
  if (argc >= 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  printf("Matrix multiplication  A(%d×%d) × B(%d×%d) = C(%d×%d)\n", M, N, N, K,
         M, K);

  DTYPE *A = alloc_matrix(M, N);
  DTYPE *B = alloc_matrix(N, K);
  DTYPE *C1 = alloc_matrix(M, K);
  DTYPE *C2 = alloc_matrix(M, K);
  DTYPE *C3 = alloc_matrix(M, K);
  DTYPE *C4 = alloc_matrix(M, K);
  DTYPE *C5 = alloc_matrix(M, K);

  init_matrix(A, M, N, 0.0);
  init_matrix(B, N, K, 0.5);

  double t0 = now();
  matmul_serial(A, B, C1, M, N, K);
  double t1 = now();
  printf("1. Serial             : %.4f s  (reference)\n", t1 - t0);

  t0 = now();
  matmul_omp_parallel(A, B, C2, M, N, K);
  t1 = now();
  printf("2. OMP parallel       : %.4f s correct=%s\n", t1 - t0,
         verify(C1, C2, M, K) ? "YES" : "NO!");

  t0 = now();
  matmul_omp_simd(A, B, C3, M, N, K);
  t1 = now();
  printf("3. OMP SIMD           : %.4f s correct=%s\n", t1 - t0,
         verify(C1, C3, M, K) ? "YES" : "NO!");

  t0 = now();
  matmul_omp_hybrid(A, B, C4, M, N, K);
  t1 = now();
  printf("4. OMP Hybrid         : %.4f s correct=%s\n", t1 - t0,
         verify(C1, C4, M, K) ? "YES" : "NO!");

  t0 = now();
  matmul_omp_gpu(A, B, C5, M, N, K);
  t1 = now();
  printf("5. OMP GPU            : %.4f s correct=%s\n", t1 - t0,
         verify(C1, C5, M, K) ? "YES" : "NO!");

  free(A);
  free(B);
  free(C1);
  free(C2);
  free(C3);
  free(C4);
  free(C5);
  return 0;
}
