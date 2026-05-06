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