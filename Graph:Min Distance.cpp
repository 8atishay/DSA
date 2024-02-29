
/*
Dijkstra Algo
1. Finding shortest path from one scource with non-negative edge weights
2. Time: O(V^2)
3. Space: O((V + E) log V)
Dis(a, b) = distance between a and b
adj: vector(vector(edges)) // a'th element: vector of {b, dis(a,b)}
*/
void dijkstra(vector<int> &distance, vector<vector<pair<int, int>>>&adj, int start) {
    int vertices = adj.size();
    distance = vector<int>(vertices, 100000000000); //not INT_MAX to save from overflow
    distance[start] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>> > pq;
    pq.push({0, start});

    while (!pq.empty()) {
        int curr = pq.top().second;  // taking shortest distance vertex from source

        pq.pop();
        for (auto edge : adj[curr]) { // edge connected to curr, edge is pair {b, dis(curr, b)}
            if (distance[edge.first] > distance[curr] + edge.second) { // Dis(start, b) > Dis(start, curr) + Dis(curr, b)
                distance[edge.first] = distance[curr] + edge.second; // update distance
                pq.push({distance[edge.first], edge.first}); // add edge for next traversal
            }
        }
    }
}


/*
Bellman-Ford Algo
1. Finding shortest path from one scource,
    works for negative weights but not for negative cycle
2. Time: O(V*E)
3. Space: O(V*E)
Dis(a, b) = distance between a and b
adj: vector(vector(edges)) // a'th element: vector of {b, dis(a,b)}
*/
void bellmanFord(vector<int> &distance, vector<vector<pair<int, int>>>&adj, int start) {
    int vertices = adj.size();
    distance = vector<int>(vertices, 10000000000); //not INT_MAX to save from overflow
    distance[start] = 0;

    int repeat = vertices - 1; //have to repeat for (vertices-1) time that is Bellman-Ford Algo
    for (int i = 0; i < repeat; i++) {
        for (int vertex = 0; vertex < vertices; vertex++) { // all the vertices in graph
            for (auto edge : adj[vertex]) { // edge originating from vertex, edge is pair {b, dis(curr, b)}
                if (distance[edge.first] > distance[vertex] + edge.second) { // Dis(start, b)> Dis(start, vertex) + Dis(vertex, b)
                    //check if there exist a shorter path to edge.first passing from vertex
                    distance[edge.first] = distance[vertex] + edge.second; // update distance
                }
            }
        }
    }

    for (int vertex = 0; vertex < vertices - 1; vertex++) { // have to check negative cycle for all edges
        for (auto edge : adj[vertex]) {
            if (distance[vertex] + edge.second < distance[edge.first])  { // if there still exists a shorter path then there must be a negative cycle
                printf("Graph contains negative weight cycle");
                return;
            }
        }
    }
}

/*
Floyd-Warshall Algo
1. Finding all pairs of shortest path,
    works for negative weights but not for negative cycle
2. Time: O(n^3)
3. Space: O(n^2)
*/
void floydWarshall(int graph[][V]) {
    for (int k = 0; k < V; k++) { // for each vertex pair {i, j} we check if there exist a shorter path passing from k
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }
}


/*
Johnson's Algo: {In my view it is Dijkstra's for negitive weights}
1. Finding all pairs of shortest path,
    works for negative weights but not for negative cycle
2. Time: O(V2log V + VE)

Steps:
I. Add a new vertex s to the graph, add edges from the new vertex to all vertices of G. Let the modified graph be G’.
II. Run the Bellman-Ford algorithm on G’ with s as the source.
    Let the distances calculated by Bellman-Ford be h[0], h[1], .. h[V-1].
    If we find a negative weight cycle, then return.
III. Reweight the edges of the original graph. For each edge (u, v), assign the new weight as “original weight + h[u] – h[v]”.
IV. Remove the added vertex s and run Dijkstra’s algorithm for every vertex.
*/


vector<vector<pair<int, int>>> getAdj(struct Graph* graph) {
    int V = graph->V;
    int E = graph->E;
    vector<vector<pair<int, int>>> adj(V);
    for (int i = 0; i < E; i++) {
        auto edge =  graph->edge[i];
        adj[edge.src].push_back({edge.dest, edge.weight});
    }
    return adj;
}

vector<vector<pair<int, int>>> getAdj(int graph[][V], int V) {
    vector<vector<pair<int, int>>> adj(V);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (graph[i][j] > 0) {
                adj[i].push_back({j, graph[i][j]});
            }
        }
    }
    return adj;
}







































