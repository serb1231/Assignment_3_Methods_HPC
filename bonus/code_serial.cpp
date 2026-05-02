#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#define NEURONS 1000
#define STEPS 500
#define THRESHOLD 50.0

double potentials[NEURONS];
int firings[NEURONS];

void simulate() {
    FILE *f = fopen("neuron_output.txt", "w");

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < NEURONS; i++) {
            potentials[i] += rand() % 10;
            if (potentials[i] > THRESHOLD) {
                firings[i]++;
                potentials[i] = 0;  // Reset potential
            }
            fprintf(f, "%d %d %f\n", step, i, potentials[i]);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("Simulation completed in %f seconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0);
    fclose(f);
}

int main() {
    srand(time(NULL));
    for (int i = 0; i < NEURONS; i++) {
        potentials[i] = rand() % 20;
        firings[i] = 0;
    }
    simulate();
    
    FILE *fsummary = fopen("neuron_summary.txt", "w");
    for (int i = 0; i < NEURONS; i++) {
        fprintf(fsummary, "Neuron %d fired %d times\n", i, firings[i]);
    }
    fclose(fsummary);
    
    return 0;
}
