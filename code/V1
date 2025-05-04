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

public:
    // Constructor
    SSSPTree(int vertices, int src) : V(vertices), source(src) {
        parent.resize(vertices, -1);
        distance.resize(vertices, INF);
        affected_del.resize(vertices, false);
        affected.resize(vertices, false);
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


// Algorithm 2 from the paper: Identify affected vertices (sequential version)
void identifyAffectedVertices(Graph& graph, SSSPTree& sssp, 
                             const vector<EdgeChange>& changes) {
    // Process all deletions first
    for (const auto& change : changes) {
        if (!change.isInsertion) {
            int u = change.edge.src;
            int v = change.edge.dest;
            
            // Check if this edge is part of the SSSP tree
            if (sssp.isTreeEdge(u, v)) {
                // Determine which vertex is further from the source
                int y = (sssp.getDistance(u) > sssp.getDistance(v)) ? u : v;
                
                // Mark this vertex as affected by deletion
                sssp.setDistance(y, INF);
                sssp.markAffectedByDeletion(y, true);
                sssp.markAffected(y, true);
            }
            
            // Remove edge from graph
            graph.removeEdge(u, v);
        }
    }
    
    // Process all insertions second
    for (const auto& change : changes) {
        if (change.isInsertion) {
            int u = change.edge.src;
            int v = change.edge.dest;
            double weight = change.edge.weight;
            
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

// Algorithm 3 from the paper: Update affected vertices (sequential version)
void updateAffectedVertices(const Graph& graph, SSSPTree& sssp) {
    int V = sssp.getVertexCount();
    
    // First part: Update vertices affected by deletion
    while (sssp.hasAffectedByDeletion()) {
        for (int v = 0; v < V; v++) {
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
                }
            }
        }
    }
    
    // Second part: Update distances of affected vertices
    while (sssp.hasAffected()) {
        for (int v = 0; v < V; v++) {
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
                }
            }
        }
    }
}

// Function to handle batch updates (multiple edge changes) using the two-step approach
void updateSSSPTwoStep(Graph& graph, SSSPTree& sssp, const vector<EdgeChange>& changes) {
    // Step 1: Identify affected vertices (Algorithm 2 in paper)
    identifyAffectedVertices(graph, sssp, changes);
    
    // Step 2: Update affected vertices (Algorithm 3 in paper)
    updateAffectedVertices(graph, sssp);
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

// Main function to run the SSSP update algorithm with METIS input
void main_sssp_update(const string& graphFilePath, const string& changesFilePath, 
                    int sourceVertex = 1, const string& outputFilePath = "sssp_result.txt") {
    cout << "Loading graph from " << graphFilePath << "..." << endl;
    Graph graph = Graph::fromMetisFile(graphFilePath);
    cout << "Graph loaded with " << graph.getVertexCount() << " vertices and " 
              << graph.getEdgeCount() << " edges." << endl;
    
    // Change source vertex from 1-based to 0-based indexing
    int zeroBasedSource = sourceVertex - 1;
    if (zeroBasedSource < 0) zeroBasedSource = 0;
    
    cout << "Computing initial SSSP from source vertex " << sourceVertex << "..." << endl;
    SSSPTree sssp = computeInitialSSSP(graph, zeroBasedSource);
    
    cout << "Loading changes from " << changesFilePath << "..." << endl;
    vector<EdgeChange> changes = parseChangesFile(changesFilePath);
    cout << "Loaded " << changes.size() << " changes." << endl;
    
    cout << "Applying changes to update SSSP (using two-step approach)..." << endl;
    auto startTime = chrono::high_resolution_clock::now();
    
    // Use the two-step approach to update SSSP
    updateSSSPTwoStep(graph, sssp, changes);
    
    auto endTime = chrono::high_resolution_clock::now();
    auto executionTimeMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    if (changes.size() > 0) {
        cout << "SSSP update completed in " << executionTimeMs << " ms." << endl;
    } else {
        cout << "No changes to apply. SSSP remains unchanged." << endl;
    }
    //cout << "SSSP update completed in " << executionTimeMs << " ms." << endl;
    
    // Save results
    sssp.saveToFile(outputFilePath);
    cout << "Results saved to " << outputFilePath << endl;
}

int main() {
    // Default filenames (can be expanded to accept command line args)
    string graphFile = "graph.txt";
    string changesFile = "changes.txt";
    int sourceVertex = 1;  // 1-based indexing
    string outputFile = "sssp_result.txt";
    
    try {
        main_sssp_update(graphFile, changesFile, sourceVertex, outputFile);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
