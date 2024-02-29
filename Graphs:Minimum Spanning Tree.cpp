/*
Prims's Algo
1. A minimum spanning tree algorithm
	finds the subset of the edges of that graph which form a tree that includes every vertex
	find the minimum sum of weights among all the trees that can be formed from the graph
2. Time: O(E log V)
adj: vector(vector(edges)) // a'th element: vector of {b, dis(a,b)}
*/
int prim(vector<vector<pair<int, int>>> adj) {
	int n = adj.size();
	vector<bool> visited(n, false);
	priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>> > pq;
	//pq is {weight, destination} of edge

	int startVertex = -1;
	int minWeight = INT_MAX;
	for (int u = 0; u < n; ++u) {
		for (auto edge : adj[u]) {
			if (edge.second < minWeight) {
				startVertex = edge.first;
				minWeight = edge.second;
			}
		}
	}
	pq.push(0, startVertex);

	int ans = 0;
	while (!pq.empty()) {
		auto [weight, destination] = pq.top(); //taking smnallest weight
		pq.pop();
		if (visited[destination]) continue; // if already added continue;
		visited[destination] = true;

		ans += weight;
		for (auto edge : adj[destination]) {
			if (visited[edge.first]) continue; // if already added continue;
			pq.push({edge.second, edge.first});
		}
	}
	return ans;
}



/*
Kruskal's Algo
1. A minimum spanning tree algorithm
	finds the subset of the edges of that graph which form a tree that includes every vertex
	find the minimum sum of weights among all the trees that can be formed from the graph
2. Time: O(E * logE)
Steps:
I: Sort all the edges in non-decreasing order of their weight. 
II. Pick the smallest edge. Check if it forms a cycle with the spanning tree formed so far. If the cycle is not formed, include this edge. Else, discard it. 
III. Repeat step#2 until there are (V-1) edges in the spanning tree.

*/