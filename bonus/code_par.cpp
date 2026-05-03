#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#ifndef NEURONS
#define NEURONS 1000
#endif
#define STEPS 500
#ifndef THRESHOLD
#define THRESHOLD 50.0
#endif

double potentials[NEURONS];
int firings[NEURONS];

// added for random contention
unsigned int thread_seeds[128];

void simulate(int num_threads) {
    FILE *f = fopen("neuron_output.txt", "w");
    auto start = omp_get_wtime();

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto seed = thread_seeds[tid];
        for (int step = 0; step < STEPS; step++)
        {
            #pragma omp single
            {
                for (int i = 0; i < NEURONS; i++) {
                    #pragma omp task shared(potentials, firings, step) firstprivate(i)
                    {
                        potentials[i] += rand_r(&seed) % 10;
                        if (potentials[i] > THRESHOLD) {
                            firings[i]++;
                            potentials[i] = 0;  // Reset potential
                        }
                        // fprintf(f, "%d %d %f\n", step, i, potentials[i]);
                    }
                }
            }
        }
    }
    auto end = omp_get_wtime();
    printf("time: %f seconds threads: %d \n", end - start, num_threads);
    fclose(f);
}

int main() {
    srand(time(NULL));
    // the creation of the configuration is not going to be measured

    for (int nr_threads = 1; nr_threads <= 128; nr_threads *= 2) {
        printf("Running with %d threads...\n", nr_threads);

        for (int i = 0; i < NEURONS; i++) {
            potentials[i] = rand() % 20;
            firings[i] = 0;
        }

        simulate(nr_threads);
        

        // The writting of the output is not going to be measured
        FILE *fsummary = fopen("neuron_summary.txt", "w");
        for (int i = 0; i < NEURONS; i++) {
            fprintf(fsummary, "Neuron %d fired %d times\n", i, firings[i]);
        }
        fclose(fsummary);
    }
    
    return 0;
}
