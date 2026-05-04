/*
 * Breadth-First Search (BFS)
 *
 */

#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <omp.h>

/* Returns wall-clock time in seconds using CLOCK_MONOTONIC. */
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct Graph {
    int n;                       // number of vertices
    long long m;                 // number of directed edges
    std::vector<int> row_ptr;    // row_ptr[v]..row_ptr[v+1]-1 → neighbours of v
    std::vector<int> col_idx;    // neighbour list

    Graph() {}
    Graph(int n, long long m) : n(n), m(m), row_ptr(n + 1, 0) {}

    // Build the graph from a list of (src, dst) pairs (0-indexed, undirected)
    static Graph from_edges(int n,
                            const std::vector<std::pair<int,int>>& edges)
    {
        // Each undirected edge becomes two directed edges
        long long m2 = (long long)edges.size() * 2;
        Graph g(n, m2);

        // Count the degree of each vertex
        for (auto& [u, v] : edges) {
            g.row_ptr[u + 1]++;
            g.row_ptr[v + 1]++;
        }
        // Calculate prefix sum to determine offsets
        for (int i = 1; i <= n; i++)
            g.row_ptr[i] += g.row_ptr[i - 1];

        g.col_idx.resize(m2);
        std::vector<int> pos(g.row_ptr.begin(), g.row_ptr.end());

        for (auto& [u, v] : edges) {
            g.col_idx[pos[u]++] = v;
            g.col_idx[pos[v]++] = u;
        }
        return g;
    }

    int degree(int v) const { return row_ptr[v + 1] - row_ptr[v]; }
};

// ─────────────────────────────────────────────
//  Utilities: Print Graph
// ─────────────────────────────────────────────

void print_stats(const Graph& g, const std::vector<int>& dist, int root)
{
    long long reachable = 0, total_dist = 0, max_dist = 0;
    for (int v = 0; v < g.n; v++) {
        if (dist[v] >= 0) {
            reachable++;
            total_dist += dist[v];
            max_dist = std::max(max_dist, (long long)dist[v]);
        }
    }
    std::cout << "  Vertices Reachable form the Root "<< root <<": " << reachable << " / " << g.n << "\n";
    std::cout << "  Diameter (max dististance from Root): " << max_dist << "\n";
    if (reachable > 1)
        std::cout << "  Avg distance       : "
                  << (double)total_dist / (reachable - 1) << "\n";
}

// ─────────────────────────────────────────────
//  Utilities: 2 common Graph Generators
// ─────────────────────────────────────────────

Graph generate_random(int n, double p, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::bernoulli_distribution coin(p);
    std::vector<std::pair<int,int>> edges;

    for (int u = 0; u < n; u++)
        for (int v = u + 1; v < n; v++)
            if (coin(rng))
                edges.push_back({u, v});

    return Graph::from_edges(n, edges);

    /* Alternatively, you can implement code to save the generated graph and re-load for new runs */
}

Graph generate_scale_free(int n, int m_attach, unsigned seed = 42)
{
    assert(m_attach >= 1 && m_attach < n);
    std::mt19937 rng(seed);
    std::vector<std::pair<int,int>> edges;
    std::vector<int> degree_sum; 

    // Seed with a small clique of size m_attach+1
    for (int u = 0; u <= m_attach; u++)
        for (int v = u + 1; v <= m_attach; v++) {
            edges.push_back({u, v});
            degree_sum.push_back(u);
            degree_sum.push_back(v);
        }

    for (int new_v = m_attach + 1; new_v < n; new_v++) {
        std::vector<int> targets;
        while ((int)targets.size() < m_attach) {
            std::uniform_int_distribution<int> pick(0, (int)degree_sum.size() - 1);
            int candidate = degree_sum[pick(rng)];
            if (candidate != new_v &&
                std::find(targets.begin(), targets.end(), candidate) == targets.end())
                targets.push_back(candidate);
        }
        for (int t : targets) {
            edges.push_back({new_v, t});
            degree_sum.push_back(new_v);
            degree_sum.push_back(t);
        }
    }
    return Graph::from_edges(n, edges);
}

/* 1. naive serial Level-synchronous BFS: each level (frontier) is processed before moving on.  */

std::vector<int> bfs_serial(const Graph& g, int src)
{
    std::vector<int> dist(g.n, -1);// -1 if unreachable.
    dist[src] = 0;

    // Represent frontier as two alternating arrays 
    std::vector<int> frontier, next_frontier;
    frontier.push_back(src);

    while (!frontier.empty()) {
        next_frontier.clear();

        for (int i = 0; i < (int)frontier.size(); i++) {
            int u = frontier[i];
            for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++) {
                int v = g.col_idx[j];
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    next_frontier.push_back(v);
                }
            }
        }
        std::swap(frontier, next_frontier);
    }
    return dist;
}

/* 2. an OpenMP Parallel Level-synchronous BFS: threads share and splits frontiers */

std::vector<int> bfs_omp_parallel(const Graph& g, int src)
{
    std::vector<int> dist(g.n, -1);// -1 if unreachable.
    dist[src] = 0;

    std::vector<bool> concurrentFlag(g.n, false);
    concurrentFlag[src] = true;

    // Represent frontier as two alternating arrays
    std::vector<int> frontier, next_frontier;
    frontier.push_back(src);
    
    /* TODO: add needed data structures, OpenMP directives, thread-local next_frontiers, handle concurrent access */
    while (!frontier.empty()) {
        next_frontier.clear();
        #pragma omp parallel for schedule(dynamic) 
        for (int i = 0; i < (int)frontier.size(); i++) {
            int u = frontier[i];
            for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++) {
                int v = g.col_idx[j];
                    if (dist[v] == -1) {
                        #pragma omp critical 
                        {
                            if (!concurrentFlag[v]) {
                                concurrentFlag[v] = true;
                                dist[v] = dist[u] + 1;
                                next_frontier.push_back(v);
                            }
                        }
                    }   
                }
        }
        std::swap(frontier, next_frontier);
    }
    return dist;
}

/* 3. an OpenMP task Level-synchronous BFS */
/* one manager thread goes through frontier to spawn one worker thread per vertex */
/* each worker thread manages its local next_frontiers */

std::vector<int> bfs_omp_task(const Graph& g, int src)
{
    std::vector<int> dist(g.n, -1);
    dist[src] = 0;

    std::vector<int> frontier, next_frontier;
    frontier.push_back(src);

    int depth = 0;

    std::vector<std::vector<int>> local_frontiers(omp_get_max_threads());

    while (!frontier.empty()) {
        next_frontier.clear();

        #pragma omp parallel
        {
            #pragma omp single  
            {
                int chunk_size = (frontier.size() + omp_get_num_threads() - 1) / omp_get_num_threads();
                for (int i = 0; i < (int)frontier.size(); i+= chunk_size) {
                    int start = i;
                    int end = std::min(i + chunk_size, (int)frontier.size());
                    #pragma omp task firstprivate(start, end)  
                    {
                        int tid = omp_get_thread_num();
                        for (int j = start; j < end; j++) {
                            int u = frontier[j];
                            for (int k = g.row_ptr[u]; k < g.row_ptr[u + 1]; k++) {
                                int v = g.col_idx[k];
                                local_frontiers[tid].push_back(v);
                            }
                        }
                    }
                }
            }
        }

        for (auto& lf : local_frontiers) {
            for (int v : lf) {
                if (dist[v] == -1) {
                    dist[v] = depth + 1;
                    next_frontier.push_back(v);
                }
            }
            lf.clear();
        }

        depth++;
        std::swap(frontier, next_frontier);
    }
    return dist;
}


int main(int argc, const char * argv[])
{
    int g_mode = 0; // 0: scale-free, 1: random
    if(argc>1) g_mode = atoi(argv[1]);

    int n = 1000000;
    if(argc>2) n = atoi(argv[2]);

    Graph g;
    int root = n/2;

    switch(g_mode){
        case 1:
            {
                double p = 0.01;
                std::cout << "=== Random Graph (" << n << ", p=" << p << ") ===\n";
                g = generate_random(n, p);
                std::cout << "  Edges: " << g.m / 2 << "\n";
                break;
            }
        case 0:
        default:
            int m_attach = 32;
            std::cout << "=== Scale-free Graph (n=" << n << ", m=" << m_attach << ") ===\n";
            g = generate_scale_free(n, m_attach);
            std::cout << "  Edges: " << g.m / 2 << "\n";
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto dist = bfs_serial(g, root);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "  Elapsed Time Serial: " << duration.count() << "\n";
    print_stats(g, dist, root);
    std::cout << "\n";

    start_time = std::chrono::high_resolution_clock::now();
    dist = bfs_omp_parallel(g, root);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "  Elapsed Time OMP Parallel: " << duration.count() << "\n";
    print_stats(g, dist, root);
    std::cout << "\n";

    
    start_time = std::chrono::high_resolution_clock::now();
    dist = bfs_omp_task(g, root);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "  Elapsed Time OMP Task: " << duration.count() << "\n";
    print_stats(g, dist, root);
    std::cout << "\n";
    
    return 0;
}
