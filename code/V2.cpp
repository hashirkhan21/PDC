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
#include <cmath>
#include <mpi.h>
#include <random>
#include <set>

using namespace std;

// Define infinity for distance values
const double INF = numeric_limits<double>::infinity();

// Structure to represent an edge in the graph
struct Edge {
    int src;
    int dest;
    double weight;
    
    Edge(int s, int d, double w) : src(s), dest(d), weight(w) {}

    // For comparing edges (useful in sets/maps)
    bool operator==(const Edge& other) const {
        return (src == other.src && dest == other.dest && weight == other.weight);
    }
};

// Custom hash function for Edge
namespace std {
    template<>
    struct hash<Edge> {
        size_t operator()(const Edge& e) const {
            return hash<int>()(e.src) ^ hash<int>()(e.dest);
        }
    };
}

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
    
    // Add an edge to the graph
    void addEdge(int u, int v, double weight) {
        edges.push_back(Edge(u, v, weight));
        adjacencyList[u].push_back(Neighbor(v, weight));
        adjacencyList[v].push_back(Neighbor(u, weight)); // For undirected graph
    }
    
    // Remove an edge from the graph
    void removeEdge(int u, int v) {
        // Remove from edges list
        edges.erase(
            remove_if(edges.begin(), edges.end(), 
                [u, v](const Edge& e) { 
                    return (e.src == u && e.dest == v) || (e.src == v && e.dest == u); 
                }
            ),
            edges.end()
        );
        
        // Remove from adjacency list
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
        
        // Check if the graph has weights (format flag 1 or 11)
        bool hasWeights = (format == 1 || format == 11);
        
        Graph graph(numVertices);
        
        // Parse adjacency lists
        for (int i = 0; i < numVertices; i++) {
            if (file.eof()) break;
            
            getline(file, line);
            istringstream lineStream(line);
            
            if (hasWeights) {
                // If graph has weights, adjacency info alternates between vertex and weight
                int vertex;
                double weight;
                
                while (lineStream >> vertex >> weight) {
                    vertex--; // METIS uses 1-based indexing
                    
                    // Add edge only if we haven't added it before (avoid duplicates in undirected graph)
                    if (i < vertex) {
                        graph.addEdge(i, vertex, weight);
                    }
                }
            } else {
                // If graph doesn't have weights, each entry is just a vertex
                int vertex;
                
                while (lineStream >> vertex) {
                    vertex--; // METIS uses 1-based indexing
                    
                    // Add edge only if we haven't added it before (avoid duplicates in undirected graph)
                    if (i < vertex) {
                        graph.addEdge(i, vertex, 1.0); // Default weight is 1
                    }
                }
            }
        }
        
        file.close();
        return graph;
    }

    // Create subgraph for a processor by partitioning vertices
    Graph createPartition(const vector<int>& partitionVertices) const {
        Graph subgraph(V);  // We keep the same vertex IDs for simplicity
        
        // Add edges where both endpoints are in the partition
        for (int v : partitionVertices) {
            for (const Neighbor& neighbor : adjacencyList[v]) {
                if (find(partitionVertices.begin(), partitionVertices.end(), neighbor.vertex) != partitionVertices.end()) {
                    if (v < neighbor.vertex) {  // Avoid adding edges twice
                        subgraph.addEdge(v, neighbor.vertex, neighbor.weight);
                    }
                }
            }
        }
        
        return subgraph;
    }

    // Add boundary edges - edges that cross partitions
    void addBoundaryEdges(const vector<int>& partitionVertices, const Graph& originalGraph) {
        set<pair<int, int>> existingEdges;
        
        // Create a set of existing edges in the current subgraph
        for (const Edge& e : edges) {
            existingEdges.insert({min(e.src, e.dest), max(e.src, e.dest)});
        }
        
        // Add boundary edges - edges from partition vertices to outside
        for (int v : partitionVertices) {
            for (const Neighbor& neighbor : originalGraph.adjacencyList[v]) {
                int u = neighbor.vertex;
                pair<int, int> edgePair = {min(v, u), max(v, u)};
                
                // If this edge doesn't exist in the subgraph, add it
                if (existingEdges.find(edgePair) == existingEdges.end()) {
                    addEdge(v, u, neighbor.weight);
                }
            }
        }
    }
};

// Class to represent the SSSP tree
class SSSPTree {
private:
    int V; // Number of vertices
    int source; // Source vertex
    vector<int> parent; // Parent of each vertex in the SSSP tree
    vector<double> distance; // Distance of each vertex from the source
    vector<bool> affected_del; // If vertex is affected by deletion
    vector<bool> affected; // If vertex is affected by any change
    vector<bool> isLocal; // Whether a vertex is local to this partition

public:
    // Constructor
    SSSPTree(int vertices, int src) : V(vertices), source(src) {
        parent.resize(vertices, -1);
        distance.resize(vertices, INF);
        affected_del.resize(vertices, false);
        affected.resize(vertices, false);
        isLocal.resize(vertices, false);
        distance[source] = 0;
    }
    
    // Check if edge (u,v) is part of the SSSP tree
    bool isTreeEdge(int u, int v) const {
        return (parent[v] == u || parent[u] == v);
    }
    
    // Get parent of a vertex
    int getParent(int vertex) const {
        return parent[vertex];
    }
    
    // Set parent of a vertex
    void setParent(int vertex, int p) {
        parent[vertex] = p;
    }
    
    // Get distance of a vertex from source
    double getDistance(int vertex) const {
        return distance[vertex];
    }
    
    // Set distance of a vertex from source
    void setDistance(int vertex, double dist) {
        distance[vertex] = dist;
    }
    
    // Mark vertex as affected by deletion
    void markAffectedByDeletion(int vertex, bool value) {
        affected_del[vertex] = value;
    }
    
    // Check if vertex is affected by deletion
    bool isAffectedByDeletion(int vertex) const {
        return affected_del[vertex];
    }
    
    // Mark vertex as affected
    void markAffected(int vertex, bool value) {
        affected[vertex] = value;
    }
    
    // Check if vertex is affected
    bool isAffected(int vertex) const {
        return affected[vertex];
    }
    
    // Get source vertex
    int getSource() const {
        return source;
    }
    
    // Get number of vertices
    int getVertexCount() const {
        return V;
    }

    // Mark vertex as local
    void markLocal(int vertex, bool value) {
        isLocal[vertex] = value;
    }

    // Check if vertex is local
    bool isVertexLocal(int vertex) const {
        return isLocal[vertex];
    }
    
    // Check if any vertex is affected by deletion
    bool hasAffectedByDeletion() const {
        for (int i = 0; i < V; i++) {
            if (affected_del[i]) return true;
        }
        return false;
    }
    
    // Check if any vertex is affected
    bool hasAffected() const {
        for (int i = 0; i < V; i++) {
            if (affected[i]) return true;
        }
        return false;
    }
    
    // Get all children of a vertex in the SSSP tree
    vector<int> getChildren(int vertex) const {
        vector<int> children;
        for (int i = 0; i < V; i++) {
            if (parent[i] == vertex) {
                children.push_back(i);
            }
        }
        return children;
    }
    
    // Save SSSP tree to file with 1-based indexing
    void saveToFile(const string& filePath) const {
        ofstream file(filePath);
        if (!file.is_open()) {
            throw runtime_error("Could not open file for writing: " + filePath);
        }
        
        file << "# SSSP Tree from source " << (source + 1) << endl;  // Convert to 1-based indexing for output
        file << "# Vertex\tDistance\tParent" << endl;
        
        for (int i = 0; i < V; i++) {
            // Output vertex index in 1-based indexing
            file << (i + 1) << "\t";  // Convert to 1-based indexing
            
            if (distance[i] == INF) {
                file << "INF";
            } else {
                file << distance[i];
            }
            
            file << "\t";
            
            if (parent[i] == -1) {
                file << "-";
            } else {
                file << (parent[i] + 1);  // Convert parent to 1-based indexing
            }
            
            file << endl;
        }
        
        file.close();
    }

    // Serialize a subset of the SSSP tree for communication
    vector<double> serializeDistances(const vector<int>& vertices) const {
        vector<double> result;
        for (int v : vertices) {
            result.push_back(distance[v]);
        }
        return result;
    }

    vector<int> serializeParents(const vector<int>& vertices) const {
        vector<int> result;
        for (int v : vertices) {
            result.push_back(parent[v]);
        }
        return result;
    }

    vector<bool> serializeAffected(const vector<int>& vertices) const {
        vector<bool> result;
        for (int v : vertices) {
            result.push_back(affected[v]);
        }
        return result;
    }

    vector<bool> serializeAffectedDel(const vector<int>& vertices) const {
        vector<bool> result;
        for (int v : vertices) {
            result.push_back(affected_del[v]);
        }
        return result;
    }

    // Update tree with received data
    void updateFromReceived(const vector<int>& vertices, const vector<double>& distances, 
                           const vector<int>& parents, const vector<bool>& affectedFlags,
                           const vector<bool>& affectedDelFlags) {
        for (size_t i = 0; i < vertices.size(); i++) {
            int v = vertices[i];
            distance[v] = distances[i];
            parent[v] = parents[i];
            affected[v] = affectedFlags[i];
            affected_del[v] = affectedDelFlags[i];
        }
    }
};

struct PQNode {
    int vertex;
    double distance;
    
    PQNode(int v, double d) : vertex(v), distance(d) {}
    
    // Operator overloading for priority queue
    bool operator>(const PQNode& other) const {
        return distance > other.distance;
    }
};

// Compute initial SSSP using Dijkstra's algorithm
SSSPTree computeInitialSSSP(const Graph& graph, int source) {
    int V = graph.getVertexCount();
    SSSPTree sssp(V, source);
    
    // Priority queue for Dijkstra's algorithm
    priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;
    vector<bool> processed(V, false);
    
    pq.push(PQNode(source, 0));
    
    while (!pq.empty()) {
        int u = pq.top().vertex;
        pq.pop();
        
        if (processed[u]) continue;
        processed[u] = true;
        
        for (const auto& neighbor : graph.getNeighbors(u)) {
            int v = neighbor.vertex;
            double weight = neighbor.weight;
            
            // Relaxation step
            if (sssp.getDistance(v) > sssp.getDistance(u) + weight) {
                sssp.setDistance(v, sssp.getDistance(u) + weight);
                sssp.setParent(v, u);
                pq.push(PQNode(v, sssp.getDistance(v)));
            }
        }
    }
    
    return sssp;
}

// Algorithm 2 from the paper: Identify affected vertices (MPI version)
void identifyAffectedVertices(Graph& graph, SSSPTree& sssp, 
                             const vector<EdgeChange>& changes,
                             const vector<int>& localVertices) {
    // Mark local vertices
    for (int v : localVertices) {
        sssp.markLocal(v, true);
    }

    // Process all deletions first
    for (const auto& change : changes) {
        if (!change.isInsertion) {
            int u = change.edge.src;
            int v = change.edge.dest;
            
            // Check if either endpoint is local and this edge is part of the SSSP tree
            if ((sssp.isVertexLocal(u) || sssp.isVertexLocal(v)) && sssp.isTreeEdge(u, v)) {
                // Determine which vertex is further from the source
                int y = (sssp.getDistance(u) > sssp.getDistance(v)) ? u : v;
                
                // Mark this vertex as affected by deletion
                sssp.setDistance(y, INF);
                sssp.markAffectedByDeletion(y, true);
                sssp.markAffected(y, true);
            }
            
            // Remove edge from graph only if it's a local operation
            if (sssp.isVertexLocal(u) || sssp.isVertexLocal(v)) {
                graph.removeEdge(u, v);
            }
        }
    }
    
    // Process all insertions second
    for (const auto& change : changes) {
        if (change.isInsertion) {
            int u = change.edge.src;
            int v = change.edge.dest;
            double weight = change.edge.weight;
            
            // Only process if either endpoint is local
            if (sssp.isVertexLocal(u) || sssp.isVertexLocal(v)) {
                int x, y;
                if (sssp.getDistance(u) > sssp.getDistance(v)) {
                    x = v;
                    y = u;
                } else {
                    x = u;
                    y = v;
                }
                
                // Check if the inserted edge improves the distance
                if (sssp.getDistance(y) > sssp.getDistance(x) + weight) {
                    sssp.setDistance(y, sssp.getDistance(x) + weight);
                    sssp.setParent(y, x);
                    sssp.markAffected(y, true);
                }
                
                // Add edge to graph
                graph.addEdge(u, v, weight);
            }
        }
    }
}

// Algorithm 3 from the paper: Update affected vertices (MPI version)
void updateAffectedVertices(const Graph& graph, SSSPTree& sssp, const vector<int>& localVertices) {
    int V = sssp.getVertexCount();
    bool globalHasAffectedDel = true;
    bool globalHasAffected = true;
    
    // First part: Update vertices affected by deletion
    while (globalHasAffectedDel) {
        bool localHasAffectedDel = false;
        
        for (int v : localVertices) {
            if (sssp.isAffectedByDeletion(v)) {
                // Clear the affected_del flag
                sssp.markAffectedByDeletion(v, false);
                
                // Get all children of this vertex
                vector<int> children = sssp.getChildren(v);
                
                // For each child, set distance to infinity and mark as affected
                for (int c : children) {
                    sssp.setDistance(c, INF);
                    sssp.markAffectedByDeletion(c, true);
                    sssp.markAffected(c, true);
                    localHasAffectedDel = true;
                }
            }
        }
        
        // Synchronize affected_del flags across processors
        MPI_Allreduce(&localHasAffectedDel, &globalHasAffectedDel, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    }
    
    // Second part: Update distances of affected vertices
    while (globalHasAffected) {
        bool localHasAffected = false;
        
        for (int v : localVertices) {
            if (sssp.isAffected(v)) {
                // Clear the affected flag
                sssp.markAffected(v, false);
                bool recheck = false;
                
                // Check all neighbors for possible distance updates
                for (const auto& neighborInfo : graph.getNeighbors(v)) {
                    int n = neighborInfo.vertex;
                    double weight = neighborInfo.weight;
                    
                    // Check if neighbor's distance can be improved
                    if (sssp.getDistance(n) > sssp.getDistance(v) + weight) {
                        sssp.setDistance(n, sssp.getDistance(v) + weight);
                        sssp.setParent(n, v);
                        sssp.markAffected(n, true);
                        localHasAffected = true;
                    } 
                    // Check if vertex's distance can be improved through neighbor
                    else if (sssp.getDistance(v) > sssp.getDistance(n) + weight) {
                        sssp.setDistance(v, sssp.getDistance(n) + weight);
                        sssp.setParent(v, n);
                        recheck = true;  // Need to recheck this vertex
                    }
                }
                
                // If the vertex's distance was updated, mark it as affected again
                if (recheck) {
                    sssp.markAffected(v, true);
                    localHasAffected = true;
                }
            }
        }
        
        // Synchronize affected flags across processors
        MPI_Allreduce(&localHasAffected, &globalHasAffected, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    }
}

// Partition the graph using a simple approach - Round Robin
vector<vector<int>> partitionGraphRoundRobin(int numVertices, int numPartitions) {
    vector<vector<int>> partitions(numPartitions);
    
    // Assign vertices to partitions in round-robin fashion
    for (int v = 0; v < numVertices; v++) {
        partitions[v % numPartitions].push_back(v);
    }
    
    return partitions;
}

// Get shared border vertices between partitions
vector<int> getBorderVertices(const Graph& graph, const vector<int>& partition) {
    vector<int> borderVertices;
    set<int> partitionSet(partition.begin(), partition.end());
    
    // Check each vertex in the partition
    for (int v : partition) {
        // Check all neighbors
        for (const auto& neighbor : graph.getNeighbors(v)) {
            // If neighbor is not in this partition, v is a border vertex
            if (partitionSet.find(neighbor.vertex) == partitionSet.end()) {
                borderVertices.push_back(v);
                break;  // Once we know it's a border vertex, we can stop checking
            }
        }
    }
    
    return borderVertices;
}

// Function to synchronize SSSPTree across processors
void synchronizeSSSPTree(SSSPTree& sssp, const vector<int>& borderVertices, int rank, int numProcesses) {
    int V = sssp.getVertexCount();
    
    // For each processor
    for (int p = 0; p < numProcesses; p++) {
        // Get border vertices data
        vector<double> distances = sssp.serializeDistances(borderVertices);
        vector<int> parents = sssp.serializeParents(borderVertices);
        vector<bool> affectedFlags = sssp.serializeAffected(borderVertices);
        vector<bool> affectedDelFlags = sssp.serializeAffectedDel(borderVertices);
        
        // Convert vector<bool> to vector<int> for MPI communication
        vector<int> affectedInts(affectedFlags.begin(), affectedFlags.end());
        vector<int> affectedDelInts(affectedDelFlags.begin(), affectedDelFlags.end());
        
        // Gather sizes of arrays from all processes
        int borderSize = borderVertices.size();
        vector<int> allSizes(numProcesses);
        MPI_Allgather(&borderSize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Calculate displacements for gather operations
        vector<int> displacements(numProcesses, 0);
        for (int i = 1; i < numProcesses; i++) {
            displacements[i] = displacements[i-1] + allSizes[i-1];
        }
        
        // Prepare receive buffers
        int totalSize = 0;
        for (int size : allSizes) totalSize += size;
        
        vector<int> allVertices(totalSize);
        vector<double> allDistances(totalSize);
        vector<int> allParents(totalSize);
        vector<int> allAffected(totalSize);
        vector<int> allAffectedDel(totalSize);
        
        // Gather data from all processes
        MPI_Allgatherv(borderVertices.data(), borderSize, MPI_INT, 
                     allVertices.data(), allSizes.data(), displacements.data(), 
                     MPI_INT, MPI_COMM_WORLD);
        
        MPI_Allgatherv(distances.data(), borderSize, MPI_DOUBLE, 
                     allDistances.data(), allSizes.data(), displacements.data(), 
                     MPI_DOUBLE, MPI_COMM_WORLD);
        
        MPI_Allgatherv(parents.data(), borderSize, MPI_INT, 
                     allParents.data(), allSizes.data(), displacements.data(), 
                     MPI_INT, MPI_COMM_WORLD);
        
        MPI_Allgatherv(affectedInts.data(), borderSize, MPI_INT, 
                     allAffected.data(), allSizes.data(), displacements.data(), 
                     MPI_INT, MPI_COMM_WORLD);
        
        MPI_Allgatherv(affectedDelInts.data(), borderSize, MPI_INT, 
                     allAffectedDel.data(), allSizes.data(), displacements.data(), 
                     MPI_INT, MPI_COMM_WORLD);
        
        // Convert back to bool
        vector<bool> allAffectedBool(allAffected.begin(), allAffected.end());
        vector<bool> allAffectedDelBool(allAffectedDel.begin(), allAffectedDel.end());
        
        // Update SSSP tree with received data
        sssp.updateFromReceived(allVertices, allDistances, allParents, allAffectedBool, allAffectedDelBool);
    }
}

// Function to handle batch updates (multiple edge changes) using the MPI-based two-step approach
void updateSSSPTwoStepMPI(Graph& graph, SSSPTree& sssp, const vector<EdgeChange>& changes, 
                         const vector<int>& localVertices, const vector<int>& borderVertices,
                         int rank, int numProcesses) {
    // Step 1: Identify affected vertices (Algorithm 2 in paper)
    identifyAffectedVertices(graph, sssp, changes, localVertices);
    
    // Synchronize affected vertices across processors
    synchronizeSSSPTree(sssp, borderVertices, rank, numProcesses);
    
    // Step 2: Update affected vertices (Algorithm 3 in paper)
    updateAffectedVertices(graph, sssp, localVertices);
    
    // Final synchronization of the updated SSSP tree
    synchronizeSSSPTree(sssp, borderVertices, rank, numProcesses);
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
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        istringstream lineStream(line);
        string operation;
        int u, v;
        double weight = 1.0;
        
        lineStream >> operation >> u >> v;
        if (!lineStream.eof()) {
            lineStream >> weight;
        }
        
        // Convert to 0-based indexing
        u--;
        v--;
        
        // Determine if this is an insertion or deletion
        bool isInsertion = (operation == "a" || operation == "add" || operation == "i" || operation == "insert");
        
        changes.push_back(EdgeChange(u, v, weight, isInsertion));
    }
    
    file.close();
    return changes;
}

// Main function to run the SSSP update algorithm with METIS input and MPI
void main_sssp_update_mpi(const string& graphFilePath, const string& changesFilePath, 
                    int sourceVertex = 1, const string& outputFilePath = "sssp_result.txt") {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Master process loads the graph and broadcasts it
    Graph graph(0);  // Create empty graph initially
    int V = 0;
    
    if (rank == 0) {
        cout << "Loading graph from " << graphFilePath << "..." << endl;
        auto startTime = chrono::high_resolution_clock::now();
        
        graph = Graph::fromMetisFile(graphFilePath);
        V = graph.getVertexCount();
        
        auto endTime = chrono::high_resolution_clock::now();
        auto loadTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
        cout << "Graph loaded with " << V << " vertices and " 
              << graph.getEdgeCount() << " edges in " << loadTime << " ms." << endl;
    }
    
    // Broadcast total number of vertices
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Change source vertex from 1-based to 0-based indexing
    int zeroBasedSource = sourceVertex - 1;
    if (zeroBasedSource < 0) zeroBasedSource = 0;
    
    // Create graph partitions
    vector<vector<int>> partitions = partitionGraphRoundRobin(V, size);
    
    // Each process builds its own partition
    vector<int> localVertices = partitions[rank];
    
    // Serialize the edges from the master process
    vector<Edge> allEdges;
    int edgeCount = 0;
    
    if (rank == 0) {
        allEdges = graph.getEdges();
        edgeCount = allEdges.size();
    }
    
    // Broadcast edge count
    MPI_Bcast(&edgeCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Prepare buffer for edge data
    struct EdgeData {
        int src;
        int dest;
        double weight;
    };
    
    vector<EdgeData> edgeData(edgeCount);
    
    if (rank == 0) {
        for (int i = 0; i < edgeCount; i++) {
            edgeData[i].src = allEdges[i].src;
            edgeData[i].dest = allEdges[i].dest;
            edgeData[i].weight = allEdges[i].weight;
        }
    }
    
    // Broadcast edge data
    MPI_Bcast(edgeData.data(), edgeCount * sizeof(EdgeData), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Each process builds its local graph
    Graph localGraph(V);
    for (int i = 0; i < edgeCount; i++) {
        int u = edgeData[i].src;
        int v = edgeData[i].dest;
        double w = edgeData[i].weight;
        
        // Add edge if either endpoint is in this partition
        if (find(localVertices.begin(), localVertices.end(), u) != localVertices.end() ||
            find(localVertices.begin(), localVertices.end(), v) != localVertices.end()) {
            localGraph.addEdge(u, v, w);
        }
    }
    
    if (rank == 0) {
        cout << "Computing initial SSSP..." << endl;
    }
    
    // Compute initial SSSP tree
    auto startTime = chrono::high_resolution_clock::now();
    SSSPTree sssp(V, zeroBasedSource);
    
    // Master process computes the initial SSSP
    if (rank == 0) {
        sssp = computeInitialSSSP(graph, zeroBasedSource);
    }
    
    // Broadcast initial distances and parents
    vector<double> initialDistances(V);
    vector<int> initialParents(V);
    
    if (rank == 0) {
        for (int i = 0; i < V; i++) {
            initialDistances[i] = sssp.getDistance(i);
            initialParents[i] = sssp.getParent(i);
        }
    }
    
    MPI_Bcast(initialDistances.data(), V, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(initialParents.data(), V, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Update local SSSP tree with broadcasted data
    if (rank != 0) {
        for (int i = 0; i < V; i++) {
            sssp.setDistance(i, initialDistances[i]);
            sssp.setParent(i, initialParents[i]);
        }
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto initialTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    
    if (rank == 0) {
        cout << "Initial SSSP computed in " << initialTime << " ms." << endl;
        cout << "Processing changes from " << changesFilePath << "..." << endl;
    }
    
    // Load changes
    vector<EdgeChange> changes;
    
    if (rank == 0) {
        changes = parseChangesFile(changesFilePath);
        cout << "Loaded " << changes.size() << " edge changes." << endl;
    }
    
    // Broadcast changes count
    int changesCount = 0;
    if (rank == 0) {
        changesCount = changes.size();
    }
    MPI_Bcast(&changesCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Prepare buffer for change data
    struct ChangeData {
        int src;
        int dest;
        double weight;
        bool isInsertion;
    };
    
    vector<ChangeData> changeData(changesCount);
    
    if (rank == 0) {
        for (int i = 0; i < changesCount; i++) {
            changeData[i].src = changes[i].edge.src;
            changeData[i].dest = changes[i].edge.dest;
            changeData[i].weight = changes[i].edge.weight;
            changeData[i].isInsertion = changes[i].isInsertion;
        }
    }
    
    // Broadcast change data
    MPI_Bcast(changeData.data(), changesCount * sizeof(ChangeData), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Convert back to EdgeChange objects
    if (rank != 0) {
        changes.clear();
        for (int i = 0; i < changesCount; i++) {
            changes.push_back(EdgeChange(
                changeData[i].src, 
                changeData[i].dest, 
                changeData[i].weight, 
                changeData[i].isInsertion
            ));
        }
    }
    
    // Identify border vertices for each partition
    vector<int> borderVertices = getBorderVertices(localGraph, localVertices);
    
    if (rank == 0) {
        cout << "Updating SSSP tree with changes..." << endl;
    }
    
    // Start timing the update algorithm
    startTime = chrono::high_resolution_clock::now();
    
    // Update SSSP tree with changes
    updateSSSPTwoStepMPI(localGraph, sssp, changes, localVertices, borderVertices, rank, size);
    
    endTime = chrono::high_resolution_clock::now();
    auto updateTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    
    if (rank == 0) {
        cout << "SSSP tree updated in " << updateTime << " ms." << endl;
        
        // Save results to file
        sssp.saveToFile(outputFilePath);
        cout << "Results saved to " << outputFilePath << endl;
    }
}

// Main entry point
int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    try {
        string graphFilePath, changesFilePath, outputFilePath;
        int sourceVertex = 1;
        
        // Parse command line arguments
        if (argc < 3) {
            if (rank == 0) {
                cout << "Usage: " << argv[0] << " <graph_file> <changes_file> [source_vertex=1] [output_file=sssp_result.txt]" << endl;
                cout << "  graph_file: Path to METIS format graph file" << endl;
                cout << "  changes_file: Path to edge changes file" << endl;
                cout << "  source_vertex: Source vertex for SSSP (1-based indexing, default=1)" << endl;
                cout << "  output_file: Path to output file (default=sssp_result.txt)" << endl;
            }
            MPI_Finalize();
            return 1;
        }
        
        graphFilePath = argv[1];
        changesFilePath = argv[2];
        
        if (argc >= 4) {
            sourceVertex = stoi(argv[3]);
        }
        
        if (argc >= 5) {
            outputFilePath = argv[4];
        } else {
            outputFilePath = "sssp_result.txt";
        }
        
        // Run the main algorithm
        main_sssp_update_mpi(graphFilePath, changesFilePath, sourceVertex, outputFilePath);
        
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error: " << e.what() << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
