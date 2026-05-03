import matplotlib.pyplot as plt
import pandas as pd
import io

csv_data = """threads,time_seconds
1,0.983707
2,0.487687
4,0.300952
8,0.125116
16,0.127712
32,0.072661
64,0.101159
128,0.453738"""

df = pd.read_csv(io.StringIO(csv_data))

# compute the speedup
baseline_time = df.iloc[0]['time_seconds']
df['speedup'] = baseline_time / df['time_seconds']

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