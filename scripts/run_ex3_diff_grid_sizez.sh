#!/bin/bash

for size in {100..1000..100}
do
    echo "Testing size: $size"
    g++ -fopenmp -DN=$size ex3/code_par.cpp -o code_par.out
    ./code_par.out $size
done