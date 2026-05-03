#!/bin/bash

for size in {1000..10000..1000}
do
    echo "Testing size: $size"
    g++ -fopenmp -DNEURONS=$size bonus/code_par.cpp -o a.out
    ./a.out $size
done