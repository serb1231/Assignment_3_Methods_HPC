#!/bin/bash

# Define the file where you want to save the results
OUTPUT_FILE="scaling_results_school.txt"

# Clear the file before starting (optional)
echo "--- Matrix Multiplication Scaling Results ---" > $OUTPUT_FILE

for threads in 1 2 4 8 16 32 64 128 256 512 1024; do
    echo "Running with $threads threads..." | tee -a $OUTPUT_FILE
    
    # Run the program and append (>>) the output to the file
    OMP_NUM_THREADS=$threads ./matmul >> $OUTPUT_FILE
    
    # Add a blank line for readability
    echo "" >> $OUTPUT_FILE
done

echo "Done! Results saved to $OUTPUT_FILE"
