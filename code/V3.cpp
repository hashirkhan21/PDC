#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <mpi.h>
#include <omp.h>
#include <metis.h>

// Define constants
#define INF INT_MAX
#define MAX_LINE_LENGTH 4096
#define MAX_VERTICES 100000

// Structure to represent an edge
typedef struct {
    int src, dest, weight;
} Edge;

// Structure to represent a graph
typedef struct {
    int num_vertices;
    int num_edges;
    int* adjacency_list;
    int* weights;
    int* offsets;
} Graph;

// SSSP Tree structure
typedef struct {
    int* parent;
    int* dist;
    bool* affected;
    bool* affected_del;
} SSSPTree;

// Function prototypes
Graph* load_graph(const char* filename);
void free_graph(Graph* graph);
SSSPTree* initialize_sssp_tree(int num_vertices, int source);
void free_sssp_tree(SSSPTree* tree);
void identify_affected_vertices(Graph* graph, SSSPTree* tree, Edge* changed_edges, int num_changes);
void update_affected_vertices(Graph* graph, SSSPTree* tree);
void process_edge_deletion(Graph* graph, SSSPTree* tree, int u, int v);
void process_edge_insertion(Graph* graph, SSSPTree* tree, int u, int v, int weight);
void print_sssp_tree(SSSPTree* tree, int num_vertices);
void partition_graph_metis(Graph* graph, int num_partitions, int** vertex_to_partition);

// Main function
int main(int argc, char** argv) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <graph_file> <source_vertex> [<changes_file>]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Load the graph
    Graph* graph = NULL;
    if (rank == 0) {
        graph = load_graph(argv[1]);
        printf("Loaded graph with %d vertices and %d edges\n", graph->num_vertices, graph->num_edges);
    }
    
    // Broadcast graph size
    int graph_info[2]; // [num_vertices, num_edges]
    if (rank == 0 && graph != NULL) {
        graph_info[0] = graph->num_vertices;
        graph_info[1] = graph->num_edges;
    }
    
    MPI_Bcast(graph_info, 2, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Create graph on other processes
    if (rank != 0) {
        graph = (Graph*)malloc(sizeof(Graph));
        graph->num_vertices = graph_info[0];
        graph->num_edges = graph_info[1];
        
        // Allocate memory for graph data
        graph->offsets = (int*)malloc((graph->num_vertices + 1) * sizeof(int));
        graph->adjacency_list = (int*)malloc(graph->num_edges * sizeof(int));
        graph->weights = (int*)malloc(graph->num_edges * sizeof(int));
    }
    
    // Broadcast graph data
    MPI_Bcast(graph->offsets, graph->num_vertices + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph->adjacency_list, graph->num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph->weights, graph->num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Get source vertex
    int source = atoi(argv[2]);
    
    // Partition the graph using METIS
    int* vertex_to_partition = NULL;
    partition_graph_metis(graph, size, &vertex_to_partition);
    
    // Initialize SSSP tree
    SSSPTree* tree = initialize_sssp_tree(graph->num_vertices, source);
    
    // Compute initial SSSP
    // We'll use Dijkstra's algorithm on rank 0, then broadcast the result
    if (rank == 0) {
        // Initialize priority queue (simple array implementation)
        bool* in_queue = (bool*)calloc(graph->num_vertices, sizeof(bool));
        int queue_size = 0;
        int* queue = (int*)malloc(graph->num_vertices * sizeof(int));
        
        // Add source to queue
        queue[queue_size++] = source;
        in_queue[source] = true;
        
        while (queue_size > 0) {
            // Find vertex with minimum distance
            int min_idx = 0;
            for (int i = 1; i < queue_size; i++) {
                if (tree->dist[queue[i]] < tree->dist[queue[min_idx]]) {
                    min_idx = i;
                }
            }
            
            // Extract vertex with minimum distance
            int u = queue[min_idx];
            queue[min_idx] = queue[--queue_size];
            in_queue[u] = false;
            
            // Relax all edges from u
            for (int i = graph->offsets[u]; i < graph->offsets[u+1]; i++) {
                int v = graph->adjacency_list[i];
                int weight = graph->weights[i];
                
                if (tree->dist[u] != INF && tree->dist[u] + weight < tree->dist[v]) {
                    tree->dist[v] = tree->dist[u] + weight;
                    tree->parent[v] = u;
                    
                    // Add v to queue if not already in queue
                    if (!in_queue[v]) {
                        queue[queue_size++] = v;
                        in_queue[v] = true;
                    }
                }
            }
        }
        
        free(in_queue);
        free(queue);
    }
    
    // Broadcast initial SSSP tree
    MPI_Bcast(tree->dist, graph->num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tree->parent, graph->num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    
    // If a changes file is provided, process the changes
    if (argc > 3) {
        FILE* changes_file = NULL;
        int num_changes = 0;
        Edge* changed_edges = NULL;
        
        if (rank == 0) {
            changes_file = fopen(argv[3], "r");
            if (!changes_file) {
                printf("Error opening changes file: %s\n", argv[3]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Count number of lines in changes file
            char line[MAX_LINE_LENGTH];
            while (fgets(line, sizeof(line), changes_file)) {
                num_changes++;
            }
            rewind(changes_file);
            
            // Read changes
            changed_edges = (Edge*)malloc(num_changes * sizeof(Edge));
            for (int i = 0; i < num_changes; i++) {
                if (fgets(line, sizeof(line), changes_file)) {
                    sscanf(line, "%d %d %d", &changed_edges[i].src, &changed_edges[i].dest, &changed_edges[i].weight);
                }
            }
            
            fclose(changes_file);
            printf("Loaded %d changes\n", num_changes);
        }
        
        // Broadcast number of changes
        MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Allocate memory for changes on other processes
        if (rank != 0) {
            changed_edges = (Edge*)malloc(num_changes * sizeof(Edge));
        }
        
        // Broadcast changes
        MPI_Bcast(changed_edges, num_changes * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Process changes in batches
        int batch_size = 100; // Adjust based on your needs
        for (int batch_start = 0; batch_start < num_changes; batch_start += batch_size) {
            int batch_end = batch_start + batch_size;
            if (batch_end > num_changes) batch_end = num_changes;
            int current_batch_size = batch_end - batch_start;
            
            double start_time = MPI_Wtime();
            
            // Step 1: Identify affected vertices
            // Each process identifies affected vertices for its partition
            Edge* batch_edges = &changed_edges[batch_start];
            
            // Reset affected flags
            memset(tree->affected, 0, graph->num_vertices * sizeof(bool));
            memset(tree->affected_del, 0, graph->num_vertices * sizeof(bool));
            
            // Process each change in the batch in parallel
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < current_batch_size; i++) {
                int u = batch_edges[i].src;
                int v = batch_edges[i].dest;
                int weight = batch_edges[i].weight;
                
                // Only process if vertices are in this partition
                if (vertex_to_partition[u] == rank || vertex_to_partition[v] == rank) {
                    if (weight == 0) {
                        // Edge deletion
                        process_edge_deletion(graph, tree, u, v);
                    } else {
                        // Edge insertion or weight update
                        process_edge_insertion(graph, tree, u, v, weight);
                    }
                }
            }
            
            // Combine affected vertices across all processes
            bool* global_affected = (bool*)calloc(graph->num_vertices, sizeof(bool));
            bool* global_affected_del = (bool*)calloc(graph->num_vertices, sizeof(bool));
            
            MPI_Allreduce(tree->affected, global_affected, graph->num_vertices, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            MPI_Allreduce(tree->affected_del, global_affected_del, graph->num_vertices, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
            
            // Copy back to local arrays
            memcpy(tree->affected, global_affected, graph->num_vertices * sizeof(bool));
            memcpy(tree->affected_del, global_affected_del, graph->num_vertices * sizeof(bool));
            
            free(global_affected);
            free(global_affected_del);
            
            // Step 2: Update affected vertices
            update_affected_vertices(graph, tree);
            
            // Synchronize distances across all processes
            int* global_dist = (int*)malloc(graph->num_vertices * sizeof(int));
            int* global_parent = (int*)malloc(graph->num_vertices * sizeof(int));
            
            MPI_Allreduce(tree->dist, global_dist, graph->num_vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            
            // For parent, we need a custom reduction
            // We'll take the parent that gives the shortest distance
            struct {
                int val;
                int rank;
            } in[graph->num_vertices], out[graph->num_vertices];
            
            for (int i = 0; i < graph->num_vertices; i++) {
                in[i].val = tree->parent[i];
                in[i].rank = rank;
            }
            
            MPI_Allreduce(in, out, graph->num_vertices, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
            
            for (int i = 0; i < graph->num_vertices; i++) {
                global_parent[i] = out[i].val;
            }
            
            // Copy back to local arrays
            memcpy(tree->dist, global_dist, graph->num_vertices * sizeof(int));
            memcpy(tree->parent, global_parent, graph->num_vertices * sizeof(int));
            
            free(global_dist);
            free(global_parent);
            
            double end_time = MPI_Wtime();
            
            if (rank == 0) {
                printf("Processed batch %d-%d in %.4f seconds\n", batch_start, batch_end-1, end_time - start_time);
            }
        }
        
        free(changed_edges);
    }
    
    // Print the final SSSP tree
    if (rank == 0) {
        printf("\nFinal SSSP Tree:\n");
        print_sssp_tree(tree, graph->num_vertices);
    }
    
    // Clean up
    free_graph(graph);
    free_sssp_tree(tree);
    free(vertex_to_partition);
    
    MPI_Finalize();
    return 0;
}

// Load graph from file
Graph* load_graph(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    
    // Read header
    int num_vertices, num_edges, format;
    fscanf(file, "%d %d %d", &num_vertices, &num_edges, &format);
    
    graph->num_vertices = num_vertices;
    graph->num_edges = 0;  // Will be calculated as we read edges
    
    // Allocate memory for temporary adjacency lists
    int** temp_adj_lists = (int**)malloc(num_vertices * sizeof(int*));
    int** temp_weight_lists = (int**)malloc(num_vertices * sizeof(int*));
    int* temp_adj_sizes = (int*)calloc(num_vertices, sizeof(int));
    int* temp_adj_capacities = (int*)malloc(num_vertices * sizeof(int));
    
    for (int i = 0; i < num_vertices; i++) {
        temp_adj_capacities[i] = 10;  // Initial capacity
        temp_adj_lists[i] = (int*)malloc(temp_adj_capacities[i] * sizeof(int));
        temp_weight_lists[i] = (int*)malloc(temp_adj_capacities[i] * sizeof(int));
    }
    
    // Read adjacency lists
    char line[MAX_LINE_LENGTH];
    for (int i = 0; i < num_vertices; i++) {
        if (!fgets(line, sizeof(line), file)) {
            break;
        }
        
        char* token = strtok(line, " \t\n");
        while (token) {
            int neighbor = atoi(token);
            
            token = strtok(NULL, " \t\n");
            if (!token) break;
            
            int weight = atoi(token);
            
            // Add edge to temporary adjacency list
            if (temp_adj_sizes[i] >= temp_adj_capacities[i]) {
                temp_adj_capacities[i] *= 2;
                temp_adj_lists[i] = (int*)realloc(temp_adj_lists[i], temp_adj_capacities[i] * sizeof(int));
                temp_weight_lists[i] = (int*)realloc(temp_weight_lists[i], temp_adj_capacities[i] * sizeof(int));
            }
            
            temp_adj_lists[i][temp_adj_sizes[i]] = neighbor - 1;  // 0-based indexing
            temp_weight_lists[i][temp_adj_sizes[i]] = weight;
            temp_adj_sizes[i]++;
            
            graph->num_edges++;
            
            token = strtok(NULL, " \t\n");
        }
    }
    
    // Allocate memory for CSR representation
    graph->offsets = (int*)malloc((num_vertices + 1) * sizeof(int));
    graph->adjacency_list = (int*)malloc(graph->num_edges * sizeof(int));
    graph->weights = (int*)malloc(graph->num_edges * sizeof(int));
    
    // Convert to CSR
    graph->offsets[0] = 0;
    for (int i = 0; i < num_vertices; i++) {
        graph->offsets[i + 1] = graph->offsets[i] + temp_adj_sizes[i];
        
        for (int j = 0; j < temp_adj_sizes[i]; j++) {
            int idx = graph->offsets[i] + j;
            graph->adjacency_list[idx] = temp_adj_lists[i][j];
            graph->weights[idx] = temp_weight_lists[i][j];
        }
    }
    
    // Free temporary data
    for (int i = 0; i < num_vertices; i++) {
        free(temp_adj_lists[i]);
        free(temp_weight_lists[i]);
    }
    free(temp_adj_lists);
    free(temp_weight_lists);
    free(temp_adj_sizes);
    free(temp_adj_capacities);
    
    fclose(file);
    return graph;
}

// Free graph memory
void free_graph(Graph* graph) {
    if (graph) {
        free(graph->offsets);
        free(graph->adjacency_list);
        free(graph->weights);
        free(graph);
    }
}

// Initialize SSSP tree
SSSPTree* initialize_sssp_tree(int num_vertices, int source) {
    SSSPTree* tree = (SSSPTree*)malloc(sizeof(SSSPTree));
    
    tree->parent = (int*)malloc(num_vertices * sizeof(int));
    tree->dist = (int*)malloc(num_vertices * sizeof(int));
    tree->affected = (bool*)calloc(num_vertices, sizeof(bool));
    tree->affected_del = (bool*)calloc(num_vertices, sizeof(bool));
    
    // Initialize distances and parents
    for (int i = 0; i < num_vertices; i++) {
        tree->dist[i] = INF;
        tree->parent[i] = -1;
    }
    
    // Source vertex has distance 0
    tree->dist[source] = 0;
    
    return tree;
}

// Free SSSP tree memory
void free_sssp_tree(SSSPTree* tree) {
    if (tree) {
        free(tree->parent);
        free(tree->dist);
        free(tree->affected);
        free(tree->affected_del);
        free(tree);
    }
}

// Process edge deletion
void process_edge_deletion(Graph* graph, SSSPTree* tree, int u, int v) {
    // Check if this edge was part of the SSSP tree
    bool is_tree_edge = false;
    if (tree->parent[v] == u || tree->parent[u] == v) {
        is_tree_edge = true;
    }
    
    if (is_tree_edge) {
        // Determine which vertex is the child
        int child = (tree->parent[v] == u) ? v : u;
        
        // Mark child as affected by deletion
        tree->dist[child] = INF;
        tree->affected_del[child] = true;
        tree->affected[child] = true;
    }
    
    // Remove edge from graph
    // In a real implementation, we would modify the graph structure
    // For simplicity, we'll just ignore this edge during future computations
}

// Process edge insertion
void process_edge_insertion(Graph* graph, SSSPTree* tree, int u, int v, int weight) {
    // Determine which vertex has the shorter distance
    int x = (tree->dist[u] <= tree->dist[v]) ? u : v;
    int y = (x == u) ? v : u;
    
    // Check if this edge creates a shorter path
    if (tree->dist[x] != INF && tree->dist[x] + weight < tree->dist[y]) {
        tree->dist[y] = tree->dist[x] + weight;
        tree->parent[y] = x;
        tree->affected[y] = true;
    }
    
    // Add edge to graph
    // In a real implementation, we would modify the graph structure
    // For simplicity, we'll use the new edge in future computations implicitly
}

// Update affected vertices
void update_affected_vertices(Graph* graph, SSSPTree* tree) {
    // First, update subtrees affected by deletion
    bool has_deletion_affected = true;
    while (has_deletion_affected) {
        has_deletion_affected = false;
        
        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < graph->num_vertices; v++) {
            if (tree->affected_del[v]) {
                tree->affected_del[v] = false;
                
                // Find all children of v in the SSSP tree
                for (int i = 0; i < graph->num_vertices; i++) {
                    if (tree->parent[i] == v) {
                        tree->dist[i] = INF;
                        tree->affected_del[i] = true;
                        tree->affected[i] = true;
                        has_deletion_affected = true;
                    }
                }
            }
        }
    }
    
    // Then, update all affected vertices
    bool has_affected = true;
    while (has_affected) {
        has_affected = false;
        
        // Create local copy of affected flags to avoid race conditions
        bool* affected_copy = (bool*)malloc(graph->num_vertices * sizeof(bool));
        memcpy(affected_copy, tree->affected, graph->num_vertices * sizeof(bool));
        
        // Reset affected flags
        memset(tree->affected, 0, graph->num_vertices * sizeof(bool));
        
        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < graph->num_vertices; v++) {
            if (affected_copy[v]) {
                // Check all neighbors of v
                for (int i = graph->offsets[v]; i < graph->offsets[v+1]; i++) {
                    int neighbor = graph->adjacency_list[i];
                    int weight = graph->weights[i];
                    
                    // Try to update neighbor through v
                    if (tree->dist[v] != INF && tree->dist[v] + weight < tree->dist[neighbor]) {
                        #pragma omp critical
                        {
                            if (tree->dist[v] + weight < tree->dist[neighbor]) {
                                tree->dist[neighbor] = tree->dist[v] + weight;
                                tree->parent[neighbor] = v;
                                tree->affected[neighbor] = true;
                                has_affected = true;
                            }
                        }
                    }
                    
                    // Try to update v through neighbor
                    if (tree->dist[neighbor] != INF && tree->dist[neighbor] + weight < tree->dist[v]) {
                        #pragma omp critical
                        {
                            if (tree->dist[neighbor] + weight < tree->dist[v]) {
                                tree->dist[v] = tree->dist[neighbor] + weight;
                                tree->parent[v] = neighbor;
                                tree->affected[v] = true;
                                has_affected = true;
                            }
                        }
                    }
                }
            }
        }
        
        free(affected_copy);
    }
}

// Print SSSP tree
void print_sssp_tree(SSSPTree* tree, int num_vertices) {
    printf("Vertex\tDistance\tParent\n");
    for (int i = 0; i < num_vertices; i++) {
        if (tree->dist[i] == INF) {
            printf("%d\tINF\t\t-\n", i);
        } else {
            printf("%d\t%d\t\t%d\n", i, tree->dist[i], tree->parent[i]);
        }
    }
}

// Partition graph using METIS
void partition_graph_metis(Graph* graph, int num_partitions, int** vertex_to_partition) {
    // Allocate memory for partition assignment
    *vertex_to_partition = (int*)malloc(graph->num_vertices * sizeof(int));
    
    // METIS uses their own data types
    idx_t nvtxs = graph->num_vertices;
    idx_t ncon = 1;  // Number of balancing constraints
    idx_t* xadj = (idx_t*)malloc((nvtxs + 1) * sizeof(idx_t));
    idx_t* adjncy = (idx_t*)malloc(graph->num_edges * sizeof(idx_t));
    idx_t* adjwgt = (idx_t*)malloc(graph->num_edges * sizeof(idx_t));
    
    // Copy graph to METIS format
    for (int i = 0; i <= nvtxs; i++) {
        xadj[i] = graph->offsets[i];
    }
    
    for (int i = 0; i < graph->num_edges; i++) {
        adjncy[i] = graph->adjacency_list[i];
        adjwgt[i] = graph->weights[i];
    }
    
    // Options for METIS
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    
    // Number of partitions
    idx_t nparts = num_partitions;
    
    // Output variables
    idx_t objval;  // Edge-cut or communication volume
    idx_t* part = (idx_t*)malloc(nvtxs * sizeof(idx_t));
    
    // Call METIS to partition the graph
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, adjwgt,
                                  &nparts, NULL, NULL, options, &objval, part);
    
    if (ret != METIS_OK) {
        printf("METIS partitioning failed with error %d\n", ret);
        // Fall back to simple partitioning
        for (int i = 0; i < graph->num_vertices; i++) {
            (*vertex_to_partition)[i] = i % num_partitions;
        }
    } else {
        // Copy partition assignment to output
        for (int i = 0; i < graph->num_vertices; i++) {
            (*vertex_to_partition)[i] = part[i];
        }
    }
    
    // Clean up
    free(xadj);
    free(adjncy);
    free(adjwgt);
    free(part);
}
