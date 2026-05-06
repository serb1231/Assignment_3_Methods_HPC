import os
import re
import matplotlib.pyplot as plt

def parse_file(filepath):
    data = {"threads": [], "Parallel": [], "SIMD": [], "Hybrid": [], "GPU": []}
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return data

    with open(filepath, 'r') as f:
        current_thread = None
        for line in f:
            t_match = re.search(r'Running with (\d+) threads', line)
            if t_match:
                current_thread = int(t_match.group(1))
                data["threads"].append(current_thread)
            
            if current_thread is not None:
                if "2. OMP parallel" in line:
                    match = re.search(r':\s+([\d.]+)\s+s', line)
                    if match: data["Parallel"].append(float(match.group(1)))
                elif "3. OMP SIMD" in line:
                    match = re.search(r':\s+([\d.]+)\s+s', line)
                    if match: data["SIMD"].append(float(match.group(1)))
                elif "4. OMP Hybrid" in line:
                    match = re.search(r':\s+([\d.]+)\s+s', line)
                    if match: data["Hybrid"].append(float(match.group(1)))
                elif "5. OMP GPU" in line:
                    match = re.search(r':\s+([\d.]+)\s+s', line)
                    if match: data["GPU"].append(float(match.group(1)))
    return data

def plot_speedup(version_name, version_key, all_data):
    plt.figure(figsize=(10, 6))
    
    colors = {"Local": "#1f77b4", "School": "#ff7f0e", "Dardel": "#2ca02c"}
    markers = {"Local": "o", "School": "s", "Dardel": "^"}
    
    threads_used = []
    
    for system, data in all_data.items():
        if not data["threads"]:
            continue
            
        threads = data["threads"]
        if len(threads) > len(threads_used):
            threads_used = threads
            
        times = data[version_key]
        
        # Ensure times matches threads length (in case of incomplete runs)
        min_len = min(len(threads), len(times))
        plot_threads = threads[:min_len]
        plot_times = times[:min_len]
        
        if min_len > 0:
            t1 = plot_times[0]
            # Speedup = T1 / TN
            speedups = [t1 / t for t in plot_times]
            
            # Plot actual speedup
            plt.plot(plot_threads, speedups, marker=markers[system], color=colors[system], 
                     linewidth=2, label=f"{system} (Actual)")
            
            # Plot ideal speedup (Speedup = N)
            plt.plot(plot_threads, plot_threads, linestyle='--', color=colors[system], 
                     alpha=0.6, label=f"{system} (Ideal)")

    plt.title(f'Speedup: {version_name}')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (T1 / TN)')
    
    # Use log scale to clearly see scaling trends
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    
    if threads_used:
        plt.xticks(threads_used, threads_used)
        
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f'speedup_{version_key.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

def main():
    base_dir = "../../ex1"
    
    files = {
        "Local": os.path.join(base_dir, "scaling_results_local.txt"),
        "School": os.path.join(base_dir, "scaling_results_school.txt"),
        "Dardel": os.path.join(base_dir, "scaling_results_dardel.txt")
    }
    
    all_data = {}
    for system, filepath in files.items():
        print(f"Parsing data for {system}...")
        all_data[system] = parse_file(filepath)
        
    versions = {
        "OMP Parallel": "Parallel",
        "OMP SIMD": "SIMD",
        "OMP Hybrid (Parallel + SIMD)": "Hybrid",
        "OMP GPU Offloading": "GPU"
    }
    
    for v_name, v_key in versions.items():
        plot_speedup(v_name, v_key, all_data)
        
    print("\nAll speedup graphs generated successfully in the scripts/ex1/ directory!")

if __name__ == "__main__":
    main()
