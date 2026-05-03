import matplotlib.pyplot as plt
import re
import numpy as np


file_path = 'data/bonus_data_vary_neuron_size.txt'

data = {}
current_size = None

try:
    with open(file_path, 'r') as f:
        for line in f:
            # match the pattern for the data for size
            size_match = re.search(r'Testing size: (\d+)', line)
            if size_match:
                current_size = int(size_match.group(1))
                data[current_size] = {'threads': [], 'times': []}
            
            # match the pattern for time and nr of threads
            time_match = re.search(r'time: ([\d\.]+) seconds threads: (\d+)', line)
            if time_match and current_size is not None:
                data[current_size]['times'].append(float(time_match.group(1)))
                data[current_size]['threads'].append(int(time_match.group(2)))

except FileNotFoundError:
    print(f"Error: {file_path} not found. Please check the filename.")
    exit()


plt.figure(figsize=(10, 6))
sizes = sorted(data.keys())

# generate the color for the lines
grays = np.linspace(0.7, 0.0, len(sizes))

for i, size in enumerate(sizes):
    # compute the baseline and the speedup
    base_time = data[size]['times'][0]
    speedups = [base_time / t for t in data[size]['times']]
    
    # plot the data
    plt.plot(
        data[size]['threads'], 
        speedups, 
        marker='o', 
        label=f'Size: {size}', 
        color=str(grays[i]),
        linewidth=1.5
    )


plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1 Thread)')

plt.title('Neuron Simulation Speedup: The Cost of Serialization', fontsize=14)
plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup ($S = T_{base} / T_n$)', fontsize=12)

plt.xscale('log', base=2)
plt.xticks(data[sizes[0]]['threads'], data[sizes[0]]['threads'])

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(title="Problem Size")
plt.tight_layout()

plt.show()