#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <chrono>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <mpi.h>
#include <metis.h>
#include <omp.h>
#include <getopt.h>

using namespace std;

// Define infinity for distance values
const double INF = numeric_limits<double>::infinity();

// Structure to represent an edge in the graph
struct Edge {
    int src;
    int dest;
    double weight;
    
    Edge(int s, int d, double w) : src(s), dest(d), weight(w) {}
};

// Structure to represent a neighbor in adjacency list
struct Neighbor {
    int vertex;
    double weight;
    
    Neighbor(int v, double w) : vertex(v), weight(w) {}
};

// Structure to represent an edge change
struct EdgeChange {
    Edge edge;
    bool isInsertion;
    
    EdgeChange(int u, int v, double w, bool insert) 
        : edge(u, v, w), isInsertion(insert) {}
};

// Class to represent a graph using adjacency list
class Graph {
private:
    int V; // Number of vertices
    vector<vector<Neighbor>> adjacencyList;
    vector<Edge> edges;

public:
    // Constructor
    Graph(int vertices) : V(vertices) {
        adjacencyList.resize(vertices);
    }
    
    // Add an edge to the graph (thread-safe)
    void addEdge(int u, int v, double weight) {
        #pragma omp critical
        {
            edges.push_back(Edge(u, v, weight));
            adjacencyList[u].push_back(Neighbor(v, weight));
            adjacencyList[v].push_back(Neighbor(u, weight)); // For undirected graph
        }
    }
    
    // Remove an edge from the graph (thread-safe)
    void removeEdge(int u, int v) {
        #pragma omp critical
        {
            edges.erase(
                remove_if(edges.begin(), edges.end(), 
                    [u, v](const Edge& e) { 
                        return (e.src == u && e.dest == v) || (e.src == v && e.dest == u); 
                    }
                ),
                edges.end()
            );
            
            adjacencyList[u].erase(
                remove_if(adjacencyList[u].begin(), adjacencyList[u].end(),
                    [v](const Neighbor& n) { return n.vertex == v; }
                ),
                adjacencyList[u].end()
            );
            
            adjacencyList[v].erase(
                remove_if(adjacencyList[v].begin(), adjacencyList[v].end(),
                    [u](const Neighbor& n) { return n.vertex == u; }
                ),
                adjacencyList[v].end()
            );
        }
    }
    
    // Get number of vertices
    int getVertexCount() const {
        return V;
    }
    
    // Get number of edges
    int getEdgeCount() const {
        return edges.size();
    }
    
    // Get adjacency list
    const vector<Neighbor>& getNeighbors(int vertex) const {
        return adjacencyList[vertex];
    }
    
    // Get all edges
    const vector<Edge>& getEdges() const {
        return edges;
    }
    
    // Load graph from METIS format file
    static Graph fromMetisFile(const string& filePath) {
        ifstream file(filePath);
        if (!file.is_open()) {
            throw runtime_error("Could not open file: " + filePath);
        }
        
        string line;
        // Skip comment lines
        do {
            getline(file, line);
        } while (line[0] == '%' && !file.eof());
        
        // Parse header line
        istringstream headerStream(line);
        int numVertices, numEdges;
        int format = 0;
        
        headerStream >> numVertices >> numEdges;
        if (!headerStream.eof()) {
            headerStream >> format;
        }
        
        // Check if the graph has weights
        bool hasWeights = (format == 1 || format == 11);
        
        Graph graph(numVertices);
        
        // Parse adjacency lists
        for (int i = 0; i < numVertices; i++) {
            if (file.eof()) break;
            
            getline(file, line);
            istringstream lineStream(line);
            
            if (hasWeights) {
                int vertex;
                double weight;
                
                while (lineStream >> vertex >> weight) {
                    vertex--; // METIS uses 1-based indexing
                    if (i < vertex) {
                        graph.addEdge(i, vertex, weight);
                    }
                }
            } else {
                int vertex;
                
                while (lineStream >> vertex) {
                    vertex--; // METIS uses 1-based indexing
                    if (i < vertex) {
                        graph.addEdge(i, vertex, 1.0);
                    }
                }
            }
        }
        
        file.close();
        return graph;
    }
};

// Class to represent the SSSP tree
class SSSPTree {
private:
    int V; // Number of vertices
    int source; // Source vertex
    vector<int> parent; // Parent of each vertex
    vector<double> distance; // Distance from source
    vector<bool> affected_del; // Affected by deletion
    vector<bool> affected; // Affected by any change

public:
    SSSPTree(int vertices, int src) : V(vertices), source(src) {
        parent.resize(vertices, -1);
        distance.resize(vertices, INF);
        affected_del.resize(vertices, false);
        affected.resize(vertices, false);
        distance[source] = 0;
    }
    
    bool isTreeEdge(int u, int v) const {
        return (parent[v] == u || parent[u] == v);
    }
    
    int getParent(int vertex) const {
        return parent[vertex];
    }
    
    void setParent(int vertex, int p) {
        parent[vertex] = p;
    }
    
    double getDistance(int vertex) const {
        return distance[vertex];
    }
    
    void setDistance(int vertex, double dist) {
        distance[vertex] = dist;
    }
    
    void markAffectedByDeletion(int vertex, bool value) {
        affected_del[vertex] = value;
    }
    
    bool isAffectedByDeletion(int vertex) const {
        return affected_del[vertex];
    }
    
    void markAffected(int vertex, bool value) {
        affected[vertex] = value;
    }
    
    bool isAffected(int vertex) const {
        return affected[vertex];
    }
    
    int getSource() const {
        return source;
    }
    
    int getVertexCount() const {
        return V;
    }
    
    bool hasAffectedByDeletion() const {
        for (int i = 0; i < V; i++) {
            if (affected_del[i]) return true;
        }
        return false;
    }
    
    bool hasAffected() const {
        for (int i = 0; i < V; i++) {
            if (affected[i]) return true;
        }
        return false;
    }
    
    vector<int> getChildren(int vertex) const {
        vector<int> children;
        for (int i = 0; i < V; i++) {
            if (parent[i] == vertex) {
                children.push_back(i);
            }
        }
        return children;
    }
    
    void saveToFile(const string& filePath) const {
        ofstream file(filePath);
        if (!file.is_open()) {
            throw runtime_error("Could not open file for writing: " + filePath);
        }
        
        file << "# SSSP Tree from source " << (source + 1) << endl;
        file << "# Vertex\tDistance\tParent" << endl;
        
        for (int i = 0; i < V; i++) {
            file << (i + 1) << "\t";
            if (distance[i] == INF) {
                file << "INF";
            } else {
                file << distance[i];
            }
            file << "\t";
            if (parent[i] == -1) {
                file << "-";
            } else {
                file << (parent[i] + 1);
            }
            file << endl;
        }
        
        file.close();
    }
};

struct PQNode {
    int vertex;
    double distance;
    
    PQNode(int v, double d) : vertex(v), distance(d) {}
    
    bool operator>(const PQNode& other) const {
        return distance > other.distance;
    }
};

// Partition graph using METIS
void partitionGraph(const Graph& graph, int nparts, vector<int>& part) {
    idx_t nvtxs = graph.getVertexCount();
    idx_t ncon = 1;
    vector<idx_t> xadj(nvtxs + 1, 0);
    vector<idx_t> adjncy;
    vector<idx_t> part_vec(nvtxs, 0);
    
    // Build CSR format for METIS
    int adjncy_size = 0;
    for (int i = 0; i < nvtxs; i++) {
        xadj[i] = adjncy_size;
        for (const auto& neighbor : graph.getNeighbors(i)) {
            adjncy.push_back(neighbor.vertex);
            adjncy_size++;
        }
    }
    xadj[nvtxs] = adjncy_size;
    
    idx_t objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                  NULL, NULL, NULL, &nparts, NULL, NULL,
                                  options, &objval, part_vec.data());
    
    if (ret != METIS_OK) {
        throw runtime_error("METIS partitioning failed");
    }
    
    part.assign(part_vec.begin(), part_vec.end());
}

// Compute initial SSSP using parallel Dijkstra's algorithm with OpenMP
SSSPTree computeInitialSSSP(const Graph& graph, int source, const vector<int>& part, int rank, int size) {
    int V = graph.getVertexCount();
    SSSPTree sssp(V, source);
    
    priority_queue<PQNode, vector<PQNode>, greater<PQNode>> pq;
    vector<bool> processed(V, false);
    
    if (part[source] == rank) {
        pq.push(PQNode(source, 0));
    }
    
    while (true) {
        // Local computation
        while (!pq.empty()) {
            int u = pq.top().vertex;
            double dist = pq.top().distance;
            pq.pop();
            
            if (processed[u]) continue;
            processed[u] = true;
            
            if (sssp.getDistance(u) > dist) {
                sssp.setDistance(u, dist);
                for (const auto& neighbor : graph.getNeighbors(u)) {
                    int v = neighbor.vertex;
                    double weight = neighbor.weight;
                    if (part[v] == rank && !processed[v]) {
                        pq.push(PQNode(v, dist + weight));
                    }
                }
            }
        }
        
        // Synchronize distances
        vector<double> send_distances(V, INF);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < V; i++) {
            if (part[i] == rank) {
                send_distances[i] = sssp.getDistance(i);
            }
        }
        
        vector<double> recv_distances(V, INF);
        MPI_Allreduce(send_distances.data(), recv_distances.data(), V, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        
        bool any_change = false;
        #pragma omp parallel for schedule(dynamic) reduction(||:any_change)
        for (int i = 0; i < V; i++) {
            if (recv_distances[i] < sssp.getDistance(i)) {
                sssp.setDistance(i, recv_distances[i]);
                if (part[i] == rank && !processed[i]) {
                    pq.push(PQNode(i, recv_distances[i]));
                    any_change = true;
                }
            }
        }
        
        int local_any_change = any_change ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &local_any_change, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!local_any_change) break;
    }
    
    return sssp;
}

// Parallel identify affected vertices with OpenMP
void identifyAffectedVertices(Graph& graph, SSSPTree& sssp, const vector<EdgeChange>& changes, const vector<int>& part, int rank, int batch_size) {
    int num_changes = changes.size();
    for (int start = 0; start < num_changes; start += batch_size) {
        int end = min(start + batch_size, num_changes);
        #pragma omp parallel for schedule(dynamic)
        for (int i = start; i < end; i++) {
            const auto& change = changes[i];
            int u = change.edge.src;
            int v = change.edge.dest;
            
            if (part[u] == rank || part[v] == rank) {
                if (!change.isInsertion) {
                    if (sssp.isTreeEdge(u, v)) {
                        int y = (sssp.getDistance(u) > sssp.getDistance(v)) ? u : v;
                        sssp.setDistance(y, INF);
                        sssp.markAffectedByDeletion(y, true);
                        sssp.markAffected(y, true);
                    }
                    graph.removeEdge(u, v);
                } else {
                    double weight = change.edge.weight;
                    int x = (sssp.getDistance(u) > sssp.getDistance(v)) ? v : u;
                    int y = (x == u) ? v : u;
                    
                    if (sssp.getDistance(y) > sssp.getDistance(x) + weight) {
                        sssp.setDistance(y, sssp.getDistance(x) + weight);
                        sssp.setParent(y, x);
                        sssp.markAffected(y, true);
                    }
                    graph.addEdge(u, v, weight);
                }
            }
        }
    }
}

// Parallel update affected vertices with OpenMP and asynchronous updates
void updateAffectedVertices(const Graph& graph, SSSPTree& sssp, const vector<int>& part, int rank, int async_level) {
    int V = sssp.getVertexCount();
    
    while (true) {
        bool any_affected = false;
        
        // Process deletion-affected vertices
        #pragma omp parallel for schedule(dynamic) reduction(||:any_affected)
        for (int v = 0; v < V; v++) {
            if (part[v] != rank) continue;
            
            if (sssp.isAffectedByDeletion(v)) {
                sssp.markAffectedByDeletion(v, false);
                queue<int> q;
                q.push(v);
                int level = 0;
                
                while (!q.empty() && level <= async_level) {
                    int x = q.front();
                    q.pop();
                    
                    vector<int> children = sssp.getChildren(x);
                    for (int c : children) {
                        sssp.setDistance(c, INF);
                        sssp.markAffectedByDeletion(c, true);
                        sssp.markAffected(c, true);
                        if (level < async_level) {
                            q.push(c);
                        }
                    }
                    level++;
                }
                any_affected = true;
            }
        }
        
        // Process affected vertices
        bool change = true;
        while (change) {
            change = false;
            #pragma omp parallel for schedule(dynamic) reduction(||:change)
            for (int v = 0; v < V; v++) {
                if (part[v] != rank) continue;
                
                if (sssp.isAffected(v)) {
                    sssp.markAffected(v, false);
                    queue<int> q;
                    q.push(v);
                    int level = 0;
                    bool local_change = false;
                    
                    while (!q.empty() && level <= async_level) {
                        int x = q.front();
                        q.pop();
                        
                        for (const auto& neighbor : graph.getNeighbors(x)) {
                            int n = neighbor.vertex;
                            double weight = neighbor.weight;
                            
                            if (sssp.getDistance(n) > sssp.getDistance(x) + weight) {
                                sssp.setDistance(n, sssp.getDistance(x) + weight);
                                sssp.setParent(n, x);
                                sssp.markAffected(n, true);
                                local_change = true;
                                if (level < async_level) {
                                    q.push(n);
                                }
                            } else if (sssp.getDistance(x) > sssp.getDistance(n) + weight) {
                                sssp.setDistance(x, sssp.getDistance(n) + weight);
                                sssp.setParent(x, n);
                                sssp.markAffected(x, true);
                                local_change = true;
                                if (level < async_level) {
                                    q.push(x);
                                }
                            }
                        }
                        level++;
                    }
                    if (local_change) {
                        change = true;
                        any_affected = true;
                    }
                }
            }
        }
        
        // Synchronize distances
        vector<double> send_distances(V, INF);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < V; i++) {
            if (part[i] == rank) {
                send_distances[i] = sssp.getDistance(i);
            }
        }
        
        vector<double> recv_distances(V, INF);
        MPI_Allreduce(send_distances.data(), recv_distances.data(), V, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        
        #pragma omp parallel for schedule(dynamic) reduction(||:any_affected)
        for (int i = 0; i < V; i++) {
            if (recv_distances[i] < sssp.getDistance(i)) {
                sssp.setDistance(i, recv_distances[i]);
                if (part[i] == rank) {
                    sssp.markAffected(i, true);
                    any_affected = true;
                }
            }
        }
        
        int local_any_affected = any_affected ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &local_any_affected, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!local_any_affected) break;
    }
}

// Parse changes from a file
vector<EdgeChange> parseChangesFile(const string& filePath) {
    ifstream file(filePath);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filePath);
    }
    
    vector<EdgeChange> changes;
    string line;
    
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        istringstream lineStream(line);
        string operation;
        int u, v;
        double weight = 1.0;
        
        lineStream >> operation >> u >> v;
        if (!lineStream.eof()) {
            lineStream >> weight;
        }
        
        u--; v--; // Convert to 0-based indexing
        bool isInsertion = (operation == "a" || operation == "add" || operation == "i" || operation == "insert");
        changes.push_back(EdgeChange(u, v, weight, isInsertion));
    }
    
    file.close();
    return changes;
}

// Log performance metrics to a file
void logPerformance(const string& logFilePath, int rank, int size, long long initialTimeMs, long long updateTimeMs, int vertexCount, int edgeCount, int changeCount, int num_threads, int batch_size, int async_level) {
    if (rank == 0) {
        ofstream logFile(logFilePath, ios::app);
        if (!logFile.is_open()) {
            cerr << "Warning: Could not open log file: " + logFilePath << endl;
            return;
        }
        
        logFile << "MPI Processes: " << size << "\n";
        logFile << "OpenMP Threads: " << num_threads << "\n";
        logFile << "Batch Size: " << batch_size << "\n";
        logFile << "Async Level: " << async_level << "\n";
        logFile << "Vertices: " << vertexCount << "\n";
        logFile << "Edges: " << edgeCount << "\n";
        logFile << "Changes: " << changeCount << "\n";
        logFile << "Initial SSSP Time (ms): " << initialTimeMs << "\n";
        logFile << "Update SSSP Time (ms): " << updateTimeMs << "\n";
        logFile << "----------------------------------------\n";
        
        logFile.close();
    }
}

// Main function for MPI+OpenMP-based SSSP update
void main_sssp_update(const string& graphFilePath, const string& changesFilePath, 
                      int sourceVertex, const string& outputFilePath, const string& logFilePath, 
                      int rank, int size, int batch_size, int async_level) {
    if (rank == 0) {
        cout << "Loading graph from " << graphFilePath << "..." << endl;
    }
    
    auto startTime = chrono::high_resolution_clock::now();
    Graph graph = Graph::fromMetisFile(graphFilePath);
    int vertexCount = graph.getVertexCount();
    int edgeCount = graph.getEdgeCount();
    
    // Partition graph using METIS
    vector<int> part;
    partitionGraph(graph, size, part);
    
    // Compute initial SSSP
    int zeroBasedSource = sourceVertex - 1;
    if (zeroBasedSource < 0) zeroBasedSource = 0;
    SSSPTree sssp = computeInitialSSSP(graph, zeroBasedSource, part, rank, size);
    
    auto endTime = chrono::high_resolution_clock::now();
    auto initialTimeMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    if (rank == 0) {
        cout << "Initial SSSP computed in " << initialTimeMs << " ms." << endl;
    }
    
    // Load and apply changes
    vector<EdgeChange> changes;
    int changeCount = 0;
    if (rank == 0) {
        cout << "Loading changes from " << changesFilePath << "..." << endl;
        changes = parseChangesFile(changesFilePath);
        changeCount = changes.size();
        cout << "Loaded " << changeCount << " changes." << endl;
    }
    
    // Broadcast changes to all processes
    int changes_size = changeCount;
    MPI_Bcast(&changes_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        changes.reserve(changes_size); // Reserve space to avoid reallocation
    }
    
    for (int i = 0; i < changes_size; i++) {
        int data[3] = {0, 0, 0};
        double weight = 0.0;
        if (rank == 0) {
            data[0] = changes[i].edge.src;
            data[1] = changes[i].edge.dest;
            data[2] = changes[i].isInsertion;
            weight = changes[i].edge.weight;
        }
        MPI_Bcast(data, 3, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            changes.emplace_back(data[0], data[1], weight, data[2]);
        }
    }
    
    if (rank == 0) {
        cout << "Applying changes to update SSSP..." << endl;
    }
    
    startTime = chrono::high_resolution_clock::now();
    identifyAffectedVertices(graph, sssp, changes, part, rank, batch_size);
    updateAffectedVertices(graph, sssp, part, rank, async_level);
    endTime = chrono::high_resolution_clock::now();
    auto updateTimeMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    
    if (rank == 0) {
        cout << "SSSP update completed in " << updateTimeMs << " ms." << endl;
        sssp.saveToFile(outputFilePath);
        cout << "Results saved to " << outputFilePath << endl;
        
        // Log performance metrics
        int num_threads = omp_get_max_threads();
        logPerformance(logFilePath, rank, size, initialTimeMs, updateTimeMs, vertexCount, edgeCount, changeCount, num_threads, batch_size, async_level);
        cout << "Performance metrics logged to " << logFilePath << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    string graphFile = "graph.txt";
    string changesFile = "changes.txt";
    int sourceVertex = 1;
    string outputFile = "sssp_result.txt";
    string logFile = "performance.log";
    int batch_size = 1000; // Default batch size
    int async_level = 2;   // Default async level
    
    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "g:c:s:o:l:b:a:")) != -1) {
        switch (opt) {
            case 'g': graphFile = optarg; break;
            case 'c': changesFile = optarg; break;
            case 's': sourceVertex = atoi(optarg); break;
            case 'o': outputFile = optarg; break;
            case 'l': logFile = optarg; break;
            case 'b': batch_size = atoi(optarg); break;
            case 'a': async_level = atoi(optarg); break;
            default:
                if (rank == 0) {
                    cerr << "Usage: " << argv[0] << " [-g graph_file] [-c changes_file] [-s source_vertex] [-o output_file] [-l log_file] [-b batch_size] [-a async_level]" << endl;
                }
                MPI_Finalize();
                return 1;
        }
    }
    
    try {
        main_sssp_update(graphFile, changesFile, sourceVertex, outputFile, logFile, rank, size, batch_size, async_level);
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error: " << e.what() << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}
