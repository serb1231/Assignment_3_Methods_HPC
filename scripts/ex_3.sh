#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J myjob
# 10 minutes wall-clock time will be given to this job
#SBATCH -t 00:10:00
#SBATCH -A edu26.DD2356
# Number of nodes
#SBATCH -p shared
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH -e error_file.e

# Run the executable file 
# and write the output into my_output_file

./run_ex3_diff_grid_sizez.sh > output_par.txt
