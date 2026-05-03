#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef N
#define N 500  // Grid size
#endif
#define ITER 1000  // Number of iterations
#define DT 0.01  // Time step
#define DX 1.0   // Grid spacing

double h[N][N], u[N][N], v[N][N];

void initialize() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            h[i][j] = 1.0;
            u[i][j] = (double)rand() / RAND_MAX;
            v[i][j] = 0.0;
        }
}

void compute(int num_threads) {
    double start = omp_get_wtime();
    omp_set_num_threads(num_threads);
    for (int iter = 0; iter < ITER; iter++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                double dudx = (u[i+1][j] - u[i-1][j]) / (2.0 * DX);
                double dvdy = (v[i][j+1] - v[i][j-1]) / (2.0 * DX);
                h[i][j] -= DT * (dudx + dvdy);
            }
        }
    }
    double end = omp_get_wtime();
    printf("Time taken: %f seconds\n", end - start);
}

void write_output() {
    FILE *f = fopen("output.txt", "w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%f ", h[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main() {
    for (int num_threads = 1; num_threads <= 128; num_threads *= 2) {
        printf("Running with %d threads.\n", num_threads);
        initialize();
        compute(num_threads);
        write_output();
    }
    printf("Computation completed.\n");
    return 0;
}