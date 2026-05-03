#!/bin/bash

for threads in 1 2 4 8 16 32; do
    echo "Running with $threads threads..."
    OMP_NUM_THREADS=$threads ./matmul
done
