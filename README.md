# ASSIGNMENT 3: Shared-Memory Programming

# Ex1 Running
For local:
```bash
gcc-15 -O3 -fopenmp hw3-omp-matmul_TBC-1.c -o matmul
./matmul.sh
```

For Dardel
```bash
salloc -N 1 -t 00:15:00 -A smio@dardel.pdc.kth.se 
cc -O3 -fopenmp hw3-omp-matmul_TBC-1.c -o matmul
sbatch ./matmul.sh
```

For School Cluster
```bash
cc -O3 -fopenmp -fno-stack-protector -fcf-protection=none hw3-omp-matmul_TBC-1.c -o matmul 
./matmul.sh
```
# Ex2 Running 

To compile the C++ code: g++ -O2 -fopenmp -o bfs.exe hw3-omp-bfs_TBC.cpp

To run the bfs.exe file - $env:OMP_NUM_THREADS={NumThreads}; .\bfs.exe  
(OMP_NUM_THREADS={NumThreads}; ./bfs.exe for linux)  

To run the bfs.exe file with a random graph - $env:OMP_NUM_THREADS={NumThreads}; .\bfs.exe 1 {NumVertices}
(OMP_NUM_THREADS={NumThreads}; ./bfs.exe 1 {NumVertices}  for linux)

# Exercise 3 Running

For exercise 3, we had 2 files: `code_par.c` and `code_ser.c`. 
For compiling and running the serial version:

```bash
gcc code_ser.c -o code_ser
./code_ser
```

For running the parallel version:

```bash
gcc -fopenmp code_par.c -o code_par
./code_par
```


# Bonus Running

For the bonus, for running the serial file one should use
```bash
g++ code_serial.cpp -o code_serial
./code_serial
```
For the parallel file, one should use
```bash
g++ -fopenmp code_par.cpp -o code_par
./code_par
```
Both of these commands will result in the file `neuron_summary.txt` being generated in the same folder. This file will contain the details regarding the neurons that fired and how much they fired. This data can be used by running
```bash
python viz_neurons.py
```
And afterwards the result can be seen and compared.

The parallel version will also print to the command line the time it took for computing for a number of threads.