# Performance Analysis of Parallel SSSP Update Algorithm on Large Graph

**Author:** Your Name  
**Date:** May 06, 2025  

## Abstract
This report presents a performance analysis of a parallel Single-Source Shortest Paths (SSSP) update algorithm implemented using MPI and OpenMP on the large `soc-Epinions1` graph dataset. The study evaluates the initial SSSP computation time and the update time for dynamic graph changes across varying configurations of MPI processes, OpenMP threads, batch sizes, and asynchronous levels. Results are analyzed for scalability and efficiency, with visualizations of speed-up performance.

## Introduction
The Single-Source Shortest Paths (SSSP) problem is a fundamental challenge in graph theory, with applications in network routing, social network analysis, and more. This report focuses on a parallel implementation of an SSSP update algorithm, extending the work of Khanda et al., using MPI for distributed computing and OpenMP for shared-memory parallelism. The large `soc-Epinions1` dataset, containing 75,888 vertices and 508,062 edges, serves as the testbed. The analysis measures initial SSSP computation time and the time to update SSSP after applying 10,000 edge changes, exploring the impact of different parallel configurations.

## Methodology
The experiment utilized a C++ implementation (`V3.cpp`) compiled with `mpicxx` and linked against METIS and OpenMP libraries. The dataset was partitioned using METIS, and the algorithm computed initial SSSP from source vertex 1, followed by dynamic updates based on changes from `changes_large.txt`. Performance metrics were logged to `performance.log` and extracted into `results.txt` using a Makefile. Configurations tested included:
- MPI Processes: 2, 4
- OpenMP Threads: 2, 4, 8
- Batch Sizes: 500, 1000, 2000
- Async Levels: 1, 2, 4
The initial SSSP time and update SSSP time were recorded in milliseconds.

## Results

### Performance Metrics
| MPI Processes | OpenMP Threads | Batch Size | Async Level | Vertices | Edges | Changes | Initial SSSP Time (ms) | Update SSSP Time (ms) |
|---------------|----------------|------------|-------------|----------|-------|---------|------------------------|-----------------------|
| 2             | 2              | 1000       | 2           | 75888    | 508062| 10000   | 2459                   | 54317                 |
| 4             | 2              | 1000       | 2           | 75888    | 508062| 10000   | 2874                   | 39031                 |
| 4             | 4              | 1000       | 2           | 75888    | 508062| 10000   | 2920                   | 81431                 |
| 4             | 8              | 1000       | 2           | 75888    | 508062| 10000   | 3027                   | 114312                |
| 4             | 4              | 500        | 2           | 75888    | 508062| 10000   | 2908                   | 91520                 |
| 4             | 4              | 2000       | 2           | 75888    | 508062| 10000   | 3092                   | 84470                 |
| 4             | 4              | 1000       | 1           | 75888    | 508062| 10000   | 2964                   | 87312                 |
| 4             | 4              | 1000       | 4           | 75888    | 508062| 10000   | 3089                   | 84752                 |

### Speed-Up Analysis
Speed-up is calculated as the ratio of the initial SSSP time with 2 MPI processes and 2 OpenMP threads (baseline: 2459 ms) to the initial SSSP time for each configuration.

| MPI Processes | OpenMP Threads | Batch Size | Async Level | Initial SSSP Time (ms) | Speed-Up |
|---------------|----------------|------------|-------------|------------------------|----------|
| 2             | 2              | 1000       | 2           | 2459                   | 1.00     |
| 4             | 2              | 1000       | 2           | 2874                   | 0.86     |
| 4             | 4              | 1000       | 2           | 2920                   | 0.84     |
| 4             | 8              | 1000       | 2           | 3027                   | 0.81     |
| 4             | 4              | 500        | 2           | 2908                   | 0.85     |
| 4             | 4              | 2000       | 2           | 3092                   | 0.80     |
| 4             | 4              | 1000       | 1           | 2964                   | 0.83     |
| 4             | 4              | 1000       | 4           | 3089                   | 0.80     |

The speed-up values indicate that increasing the number of OpenMP threads beyond 2 (with 4 MPI processes) does not improve performance and may degrade it, possibly due to overhead from thread synchronization. Similarly, varying batch sizes and async levels shows minimal impact on initial SSSP time.

### Visualization
[Note: A bar chart comparing speed-up across configurations would be inserted here. Since LaTeX is not working, you can generate this using the Python code below and convert it to an image, then insert it into a document.]

## Conclusion
The analysis reveals that the parallel SSSP update algorithm performs best with 2 MPI processes and 2 OpenMP threads for the initial SSSP computation, achieving the lowest time of 2459 ms. Increasing parallelism (e.g., 4 MPI processes with 8 threads) increases the initial SSSP time, suggesting overhead from synchronization and partitioning outweighs parallel gains. The update SSSP time varies significantly, with 4 MPI processes and 2 threads yielding the fastest update at 39031 ms, while higher thread counts (e.g., 8) lead to poorer performance (114312 ms), likely due to contention. Batch size and async level adjustments show moderate effects, with smaller batches (500) and lower async levels (1) offering slight improvements in update times. Future work could optimize thread utilization and explore adaptive partitioning strategies to enhance scalability.

## Appendix: Generating the Visualization
To create the speed-up chart, use the following Python code (requires `matplotlib` and `numpy`):
```python
import matplotlib.pyplot as plt
import numpy as np

# Data from results.txt
mpi_threads = [(2, 2), (4, 2), (4, 4), (4, 8), (4, 4), (4, 4), (4, 4), (4, 4)]
initial_times = [2459, 2874, 2920, 3027, 2908, 3092, 2964, 3089]
batch_sizes = [1000, 1000, 1000, 1000, 500, 2000, 1000, 1000]
async_levels = [2, 2, 2, 2, 2, 2, 1, 4]

# Calculate speed-up (baseline = 2459 ms)
speedup = [2459 / t for t in initial_times]

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
index = np.arange(len(mpi_threads))

bars1 = ax.bar(index, speedup, bar_width, label='Speed-Up', color='skyblue')

ax.set_xlabel('Configuration (MPI, Threads, Batch, Async)')
ax.set_ylabel('Speed-Up')
ax.set_title('Speed-Up Across Configurations')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([f'{m},{t},{b},{a}' for m, t, b, a in zip([p[0] for p in mpi_threads], [p[1] for p in mpi_threads], batch_sizes, async_levels)], rotation=45, ha='right')
ax.set_ylim(0, 1.2)
ax.legend()

plt.tight_layout()
plt.savefig('speedup_chart.png')
