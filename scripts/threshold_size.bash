#!/bin/bash

for size in {10..100..10}
do
    echo "Testing size: $size"
    g++ -fopenmp -DTHRESHOLD=$size bonus/code_par.cpp -o a.out
    ./a.out $size
done