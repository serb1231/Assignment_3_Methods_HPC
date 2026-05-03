import matplotlib.pyplot as plt

# The number of threads you tested
threads = [1, 2, 4, 8, 16, 32]

# TODO: Fill these arrays with the execution times (in seconds) 
# for the "OMP parallel" (or Hybrid) output from your terminal!
time_local = [5.0, 2.6, 1.5, 1.1, 1.0, 1.0] # Example data
time_cluster = [4.8, 2.4, 1.2, 0.7, 0.4, 0.3] # Example data
time_dardel = [4.5, 2.3, 1.1, 0.6, 0.3, 0.18] # Example data

# Calculate Ideal Scaling (based on the single-thread time of Dardel as baseline, or per-system)
# Ideal time = Time(1 thread) / number of threads
ideal_local = [time_local[0] / t for t in threads]
ideal_cluster = [time_cluster[0] / t for t in threads]
ideal_dardel = [time_dardel[0] / t for t in threads]

plt.figure(figsize=(10, 6))

# Plot actual times
plt.plot(threads, time_local, marker='o', label='Local Computer')
plt.plot(threads, time_cluster, marker='s', label='School Cluster')
plt.plot(threads, time_dardel, marker='^', label='Dardel')

# Plot ideal scaling (dashed lines)
plt.plot(threads, ideal_local, linestyle='--', color='blue', alpha=0.5, label='Ideal Local')
plt.plot(threads, ideal_cluster, linestyle='--', color='orange', alpha=0.5, label='Ideal Cluster')
plt.plot(threads, ideal_dardel, linestyle='--', color='green', alpha=0.5, label='Ideal Dardel')

plt.title('Strong Scaling: Matrix Multiplication')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (Seconds)')
plt.xticks(threads)
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('strong_scaling.png', dpi=300)
plt.show()
