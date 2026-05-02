import numpy as np
import matplotlib.pyplot as plt

# Load all data first
data = np.loadtxt("neuron_output.txt")
summary_data = np.loadtxt("neuron_summary.txt", dtype=str)
fire_counts = np.array([int(row[-2]) for row in summary_data])

# Create a figure with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# First Plot: Scatter
im = ax1.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='coolwarm', s=1)
fig.colorbar(im, ax=ax1, label='Membrane Potential')
ax1.set_title("Neuron Activity Over Time")
ax1.set_ylabel("Neuron ID")

# Second Plot: Histogram
ax2.hist(fire_counts, bins=20, color='blue', edgecolor='black')
ax2.set_title("Distribution of Neuron Firings")
ax2.set_xlabel("Firing Count")
ax2.set_ylabel("Number of Neurons")

# Adjust layout so titles don't overlap
plt.tight_layout()
plt.show()