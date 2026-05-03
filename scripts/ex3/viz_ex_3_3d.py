import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import io
import numpy as np

# open file
csv_path = 'data/ex3/pc_threads_vs_grid_size.csv'

df = pd.read_csv(csv_path)

# set up plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# plot everything
surf = ax.plot_trisurf(df['threads'], df['size'], df['time_seconds'], 
                       cmap='viridis', edgecolor='none', alpha=0.8)


# add a layer of scatter to highlight datapoints
ax.scatter(df['threads'], df['size'], df['time_seconds'], color='black', s=20)

# format it nicely
ax.set_title('Simulation Threads versus Grid Size', fontsize=15, pad=20)
ax.set_xlabel('Threads', fontsize=12)
ax.set_ylabel('Problem Size', fontsize=12)
ax.set_zlabel('Time (seconds)', fontsize=12)

# use log scale
ax.set_box_aspect([1, 1, 0.8])
ax.view_init(elev=20, azim=-135)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Execution Time')

plt.tight_layout()
plt.show()