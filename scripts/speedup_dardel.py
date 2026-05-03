import matplotlib.pyplot as plt
import pandas as pd
import io

csv_data = """threads,time
1,0.295068
2,0.905161
4,1.101895
8,1.043622
16,1.059031
32,1.055875
64,1.044552
128,1.218416"""

df = pd.read_csv(io.StringIO(csv_data))

# compute the speedup
baseline_time = df.iloc[0]['time']
df['speedup'] = baseline_time / df['time']

plt.figure(figsize=(10, 6))

plt.plot(df['threads'], df['speedup'], marker='o', linestyle='-', color='black', label='Observed Speedup')

# add ideal speedup
plt.plot(df['threads'], df['threads'], linestyle='--', color='red', alpha=0.5, label='Ideal Speedup (Linear)')

plt.title('OpenMP Scaling: Speedup vs. Number of Threads', fontsize=14)
plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup (Base Time / Parallel Time)', fontsize=12)

plt.xscale('log', base=2)
plt.xticks(df['threads'], df['threads'])

plt.ylim(0, max(df['speedup']) * 1.2) 

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()

plt.show()