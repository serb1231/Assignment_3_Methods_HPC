#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define NEURONS 1000
#define STEPS 500
#define THRESHOLD 50.0

double potentials[NEURONS];
int firings[NEURONS];

void simulate(int num_threads) {
    FILE *f = fopen("neuron_output.txt", "w");
    auto start = omp_get_wtime();

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        for (int step = 0; step < STEPS; step++)
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < NEURONS; i++) {
                {
                    potentials[i] += rand() % 10;
                    if (potentials[i] > THRESHOLD) {
                        firings[i]++;
                        potentials[i] = 0;  // Reset potential
                    }
                    fprintf(f, "%d %d %f\n", step, i, potentials[i]);
                }
            }
        }
    }
    auto end = omp_get_wtime();
    printf("Simulation completed in %f seconds for %d threads\n", end - start, num_threads);
    fclose(f);
}

int main() {
    srand(time(NULL));
    // the creation of the configuration is not going to be measured

    for (int nr_threads = 2; nr_threads <= 128; nr_threads *= 2) {
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
