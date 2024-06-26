{
// Greedy
	{
		//Easy Greedy
		{

			{
				/*
					Highest Product
					https://www.interviewbit.com/problems/highest-product/
				*/
				int Solution::maxp3(vector<int> &A) {
					sort(A.begin(), A.end());
					int n = A.size();
					int mx = max({A[n - 1]*A[n - 2]*A[n - 3], A[n - 1]*A[n - 2]*A[0], A[0]*A[1]*A[n - 1]});
					return mx;
				}
			}
			{
				/*
					Bulbs
					https://www.interviewbit.com/problems/interview-questions/
				*/
				int Solution::bulbs(vector<int> &A) {
					bool C = 0;
					int n = A.size(), count = 0;
					for (int i = 0; i < n; i++) {
						if ((C && A[i] == 1) || (!C && A[i] == 0)) {
							count++;
							C = !C;
						}
					}
					return count;
				}
			}
			{
				/*
					Disjoint Intervals
					https://www.interviewbit.com/problems/disjoint-intervals/
				*/
				int Solution::solve(vector<vector<int> > &A) {
					sort(A.begin(), A.end());
					int count = 1, i = 0, j = 1, n = A.size();
					while (j < n) {
						if (A[i][1] > A[j][1]) {
							i = j;
						}
						else if (A[i][1] < A[j][0]) {
							count++;
							i = j;
						}
						j++;
					}
					return count;
				}
			}
			{
				/*
					Largest Permutation
					https://www.interviewbit.com/problems/largest-permutation/
				*/
				vector<int> Solution::solve(vector<int> &A, int B) {
					int n = A.size();
					vector<int> v(n + 1);
					for (int i = 0; i < n; i++) {
						v[A[i]] = i;
					}
					for (int i = 0; i < n && B > 0; i++) {
						if (A[i] == n - i)continue;

						int t = v[n - i];
						v[A[i]] = v[n - i];
						v[n - i] = i;
						swap(A[t], A[i]);
						B--;
					}
					return A;
				}
			}
		}
		// Medium Greedy
		{
			{
				/*
					Meeting rooms
					https://www.interviewbit.com/problems/meeting-rooms/

					#Famous
				*/
				int Solution::solve(vector<vector<int> > &A) {
					int n = A.size();
					vector<pair<int, bool>> D;
					for (int i = 0; i < n; i++) {
						D.push_back({A[i][0], 1});
						D.push_back({A[i][1], 0});
					}
					sort(D.begin(), D.end());
					int count = 0, ans = 0;
					for (int i = 0; i < 2 * n; i++) {
						if (D[i].second)count++;
						else count--;
						ans = max(ans, count);
					}
					return ans;
				}
			}
			{
				/*
					Distribute Candy
					https://www.interviewbit.com/problems/distribute-candy/

					#Famous
				*/
				int Solution::candy(vector<int> &A) {
					int n = A.size(), count = 0;
					vector<int> Z(n, 1);
					for (int i = 1; i < n; i++) {
						if (A[i] > A[i - 1])Z[i] = Z[i - 1] + 1;
					}
					for (int i = n - 2; i >= 0; i--) {
						if (A[i] > A[i + 1])Z[i] = max(Z[i], Z[i + 1] + 1);
					}
					for (int z : Z) {
						count += z;
					}
					return count;

				}
			}
			{
				/*
				Seats
				https://www.interviewbit.com/problems/seats/
				*/

				int med(int count, int &n, string & A) {
					if (count % 2 == 0) {
						count = count / 2;
						int count2 = 0, x;
						for (int i = 0; i < n; i++) {
							if (A[i] == 'x')count2++;
							if (count2 == count) {
								x = i;
							}
							else if (count2 == count + 1) {
								return (x + i) / 2;
							}
						}
					}
					else {
						count = count / 2 + 1;
						int count2 = 0;
						for (int i = 0; i < n; i++) {
							if (A[i] == 'x')count2++;
							if (count2 == count) {
								return i;
							}
						}
					}
				}
				int Solution::seats(string A) {
					int n = A.length();
					long long count = 0;
					for (int i = 0; i < n; i++) {
						if (A[i] == 'x')count++;
					}
					if (count == 0)return 0;

					int median = med(count, n, A);
					// cout<<median<<" "<<count<<endl;
					long long ans = 0;
					for (int i = 0; i < n; i++) {
						if (A[i] == 'x') {
							ans += abs(i - median);
						}
					}
					if (count % 2 == 0) {
						count /= 2;
						return (ans - (long long)pow(count, 2)) % 10000003;
					} else {
						count /= 2;
						return (ans - count * (count + 1)) % 10000003;
					}
				}

			}
			{
				/*
				Assign Mice to Holes
				https://www.interviewbit.com/problems/assign-mice-to-holes/
				*/
				int Solution::mice(vector<int> &A, vector<int> &B) {
					sort(A.begin(), A.end());
					sort(B.begin(), B.end());
					int n = A.size(), ans = 0;
					for (int i = 0; i < n; i++) {
						ans = max(ans, abs(A[i] - B[i]));
					}
					return ans;
				}

			}
			{
				/*
				Majority Element
				https://www.interviewbit.com/problems/majority-element/

				#Famous
				*/
				int Solution::majorityElement(const vector<int> &A) {
					int ans, count = 0, n = A.size();
					for (int i = 0; i < n; i++) {
						if (count == 0) {
							ans = A[i];
							count++;
						} else {
							count += (A[i] == ans ? 1 : -1);
						}
					}
					return ans;
				}
			}
			{
				/*Gas Station
				https://www.interviewbit.com/problems/gas-station/

				#Famous
				*/
				int Solution::canCompleteCircuit(const vector<int> &A, const vector<int> &B) {
					int n = A.size(), fuel = 0, ans = 0, total = 0;
					for (int i = 0; i < n; i++) {
						fuel += (A[i] - B[i]);
						total += (A[i] - B[i]);
						if (fuel < 0) {
							fuel = 0;
							ans = i + 1;
						}
					}
					if (total >= 0)return ans;
					else return -1;
				}
			}
		}
	}
// Graph
	{
		// Graph Traversal
		{
			{
				/*
				Path in Directed Graph
				https://www.interviewbit.com/problems/path-in-directed-graph/

				#DFS #BFS #Famous
				*/
				void dfs(vector<int>g[], vector<bool>&vis, int u) {
					vis[u] = true;
					for (auto v : g[u]) {
						if (!vis[v]) {
							dfs(g, vis, v);
						}
					}
				}
				int Solution::solve(int n, vector<vector<int> > &edges) {
					vector<int>g[n + 1];
					for (auto& edge : edges) {
						g[edge[0]].push_back(edge[1]);
					}

					vector<bool>vis(n + 1, 0);

					//1.dfs - O(N), O(N)
					// dfs(g,vis,1)
					// return vis[n];

					//2.bfs - O(N), O(N)
					queue<int>q;
					q.push(1);
					vis[1] = true;
					while (q.size()) {
						int u = q.front();
						q.pop();
						for (auto v : g[u]) {
							if (v == n) return true;
							if (!vis[v]) {
								q.push(v);
								vis[v] = true;
							}
						}
					}

					return false;
				}

			}
			{
				/*
				Water Flow
				https://www.interviewbit.com/problems/water-flow/
				*/
				int n, m;
				void dfs(int i, int j, vector<vector<int>> &A, vector<vector<int>> &V) {
					if (V[i][j])return ;
					V[i][j] = 1;
					if (i > 0 && A[i - 1][j] >= A[i][j]) dfs(i - 1, j, A, V);
					if (j > 0 && A[i][j - 1] >= A[i][j]) dfs(i, j - 1, A, V);
					if (i < n - 1 && A[i + 1][j] >= A[i][j]) dfs(i + 1, j, A, V);
					if (j < m - 1 && A[i][j + 1] >= A[i][j]) dfs(i, j + 1, A, V);
				}
				int Solution::solve(vector<vector<int> > &A) {
					n = A.size();
					m = A[0].size();
					vector<vector<int>> B(n, vector<int>(m, 0)), R = B;
					for (int i = 0; i < n; i++) {
						dfs(i, 0, A, B);
						dfs(i, m - 1, A, R);
					}
					for (int j = 0; j < m; j++) {
						dfs(0, j, A, B);
						dfs(n - 1, j, A, R);
					}
					int count = 0;
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < m; j++) {
							if (B[i][j] && R[i][j])count++;
						}
					}
					return count;
				}
			}
			{
				/*
				Stepping Numbers
				https://www.interviewbit.com/problems/stepping-numbers/
				*/
				void dfs(int x, int &A, int &B, vector<int> &S) {
					if (x > B)return;
					if (A <= x && x <= B)S.push_back(x);
					if (x % 10 < 9) {
						dfs(x * 10 + x % 10 + 1, A, B, S);
					}
					if (x % 10 > 0) {
						dfs(x * 10 + x % 10 - 1, A, B, S);
					}
				}

				vector<int> Solution::stepnum(int A, int B) {
					vector<int> S;
					if (A == 0) {
						S.push_back(0);
						A++;
					}
					for (int i = 1; i <= 9; i++) {
						dfs(i, A, B, S);
					}
					sort(S.begin(), S.end());
					return S;
				}
			}
			{
				/*
				Capture Regions on Board
				https://www.interviewbit.com/problems/capture-regions-on-board/
				*/
				vector<pair<int, int>> D = {{1, 0}, { -1, 0}, {0, 1}, {0, -1}};
				void dfs(int i, int j, vector<vector<char> > &A, vector<vector<bool>> &vis, int &n, int &m) {
					if (vis[i][j])return;
					vis[i][j] = 1;
					for (auto [di, dj] : D) {
						if (i + di < n && i + di >= 0 && j + dj < m && j + dj >= 0 && A[i + di][j + dj] == 'O')
							dfs(i + di, j + dj, A, vis, n, m);
					}
				}
				void Solution::solve(vector<vector<char> > &A) {
					int n = A.size(), m = A[0].size();
					vector<vector<bool>> vis(n, vector<bool>(m, 0));
					for (int i = 0; i < n; i++) {
						if (A[i][0] == 'O') dfs(i, 0, A, vis, n, m);
						if (A[i][m - 1] == 'O') dfs(i, m - 1, A, vis, n, m);
					}
					for (int j = 0; j < m; j++) {
						if (A[0][j] == 'O') dfs(0, j, A, vis, n, m);
						if (A[n - 1][j] == 'O') dfs(n - 1, j, A, vis, n, m);
					}
					for (int i = 1; i < n - 1; i++) {
						for (int j = 1; j < m - 1; j++) {
							if (!vis[i][j]) A[i][j] = 'X';
						}
					}
				}
			}
			{
				/*
				Word Search Board
				https://www.interviewbit.com/problems/word-search-board/
				*/
				int n, m, l;
				string B;
				vector<pair<int, int>> D = {{1, 0}, { -1, 0}, {0, 1}, {0, -1}, {0, 0}};
				bool check(int ind, vector<string> &A, int a, int b) {
					if (ind == l)return 1;
					for (auto [i, j] : D) {
						i += a; j += b;
						if (i < n && i >= 0 && j < m && j >= 0 && A[i][j] == B[ind]) {
							if (check(ind + 1, A, i, j))return 1;
						}
					}
					return 0;
				}
				int Solution::exist(vector<string> &A, string Z) {
					n = A.size();
					m = A[0].size();
					B = Z;
					l = B.length();

					for (int i = 0; i < n; i++) {
						for (int j = 0; j < m; j++) {
							if (check(0, A, i, j))return 1;
						}
					}
					return 0;
				}

			}
		}
		// DFS
		{
			{
				/*
				Largest Distance between nodes of a Tree
				https://www.interviewbit.com/problems/largest-distance-between-nodes-of-a-tree/

				#Good
				*/
				int dfs(int root, vector<vector<int>> &adj, int &ans, vector<int> &dep, int &n) {
					int mdep = 0, mdep2 = 0;
					for (int x : adj[root]) {
						if (dep[x] == -1) {
							dep[x] = dfs(x, adj, ans, dep, n) + 1;
							if (dep[x] >= mdep) {
								mdep2 = mdep;
								mdep = dep[x];
							} else if (dep[x] > mdep2) {
								mdep2 = dep[x];
							}
						}
					}
					ans = max(ans, mdep + mdep2);
					return mdep;
				}
				int Solution::solve(vector<int> &A) {
					int n = A.size();
					vector<vector<int>> adj(n, vector<int>());
					int root;
					for (int i = 0; i < n; i++) {
						if (A[i] == -1) {
							root = i;
							continue;
						}
						adj[A[i]].push_back(i);
					}
					vector<int> dep(n, -1);
					int ans = 0;
					dep[root] = dfs(root, adj, ans, dep, n);
					return ans;
				}
			}
			{
				/*
				Cycle in Directed Graph
				https://www.interviewbit.com/problems/cycle-in-directed-graph/

				#Famous
				*/
				bool cycle(vector<vector<int>> &adj, vector<bool> &vis, int &A, int ind) {
					if (vis[ind])return 1;
					vis[ind] = 1;
					for (int &x : adj[ind]) {
						if (cycle(adj, vis, A, x)) return 1;
					}
					vis[ind] = 0; // Resetting to 0
					return 0;
				}
				int Solution::solve(int A, vector<vector<int> > &B) {
					vector<vector<int>> adj(A, vector<int>());
					int n = B.size();
					for (int i = 0; i < n; i++) {
						adj[B[i][0] - 1].push_back(B[i][1] - 1);
					}
					vector<bool> vis(A, false);
					for (int i = 0; i < A; i++) {
						if (cycle(adj, vis, A, i))return 1;
					}
					return 0;
				}
			}
			{
				/*
				 Delete Edge!
				 https://www.interviewbit.com/problems/delete-edge/
				*/
				typedef long long ll;
				ll MOD 1000000007;
				ll calsum(int ind, int prev, vector<ll> &sum, vector<vector<int>> &adj, vector<int> &A) {
					// if(sum[ind]!=-1)return sum[ind];
					ll s = A[ind];
					for (int i : adj[ind]) {
						if (i != prev) {
							s += calsum(i, ind, sum, adj, A);
						}
					}
					return sum[ind] = s;
				}
				int Solution::deleteEdge(vector<int> &A, vector<vector<int> > &B) {
					int nodes = A.size(), n = B.size();
					vector<vector<int>> adj(nodes, vector<int>());
					for (int i = 0; i < n; i++) {
						adj[B[i][0] - 1].push_back(B[i][1] - 1);
						adj[B[i][1] - 1].push_back(B[i][0] - 1);
					}
					vector<ll> sum(nodes, -1);
					sum[0] = calsum(0, -1, sum, adj, A);
					ll ans = INT_MIN;
					for (int i = 0; i < nodes; i++) {
						ans = max(ans, ((sum[0] - sum[i]) % MOD * sum[i] % MOD) % MOD);
					}
					return ans % MOD;
				}
			}
			{
				/*
				Two teams?
				https://www.interviewbit.com/problems/two-teams/

				#Famous #Bipartate #2Color #Good
				*/
				bool dfs(int ind, vector<bool> &color, bool col, vector<bool> &vis, vector<vector<int>> &adj) {
					if (vis[ind]) return col == color[ind];
					vis[ind] = true;
					color[ind] = col;
					for (auto x : adj[ind]) {
						if (!dfs(x, color, !col, vis, adj)) return 0;
					}
					return 1;
				}
				int Solution::solve(int A, vector<vector<int> > &B) {
					vector<vector<int>> adj(A, vector<int>());
					int n = B.size();
					for (int i = 0; i < n; i++) {
						adj[B[i][0] - 1].push_back(B[i][1] - 1);
						adj[B[i][1] - 1].push_back(B[i][0] - 1);
					}
					vector<bool> color(A), vis(A);
					for (int i = 0; i < A; i++) {
						if (vis[i]) continue;
						if (!dfs(i, color, 0, vis, adj)) return 0;
					}
					return 1;
				}
			}
		}
		// BFS
		{
			{
				/*
				Valid Path (Circle inside Rectangle)
				https://www.interviewbit.com/problems/valid-path/
				*/
				int x, y, N, R;
				vector<pair<int, int>> d = {{1, 0}, {1, 1}, {1, -1}, {0, 1}, {0, -1}, { -1, 0}, { -1, 1}, { -1, -1}};

				bool check(int a, int b, vector<int> &E, vector<int> &F) {
					for (int i = 0; i < N; i++) {
						if (pow((a - E[i]), 2) + pow((b - F[i]), 2) <= R * R)return 0;
					}
					return 1;
				}

				bool dfs(int a, int b, vector<int> &E, vector<int> &F, vector<vector<int>> &vis) {
					if (a >= 0 && a <= x && b <= y && b >= 0 && !vis[a][b] && check(a, b, E, F)) {
						if (a == x && b == y)return 1;
						vis[a][b] = 1;
						for (auto [di, dj] : d) {
							if (dfs(di + a, dj + b, E, F, vis))return true;
						}
					}
					return 0;
				}

				string Solution::solve(int A, int B, int C, int D, vector<int> &E, vector<int> &F) {
					x = A, y = B, N = C, R = D;
					vector<vector<int>> vis(A + 1, vector<int>(B + 1, 0));
					return dfs(0, 0, E, F, vis) ? "YES" : "NO";
				}
			}
			{
				/*
				Region in BinaryMatrix
				https://www.interviewbit.com/problems/region-in-binarymatrix/
				*/
				vector<pair<int, int>> D = {{1, 0}, {1, 1}, {1, -1}, {0, 1}, {0, -1}, { -1, 0}, { -1, 1}, { -1, -1}};
				void dfs(int i, int j, vector<vector<int>> &A, int &cnt) {
					if (i < 0 || j < 0 || i >= A.size() || j >= A[0].size() || A[i][j] == 0 ) return;
					cnt++;
					A[i][j] = 0;
					for (auto[di, dj] : D) {
						dfs(i + di, j + dj, A, cnt);
					}
				}
				int Solution::solve(vector<vector<int> > &A) {
					int mx = INT_MIN;
					int cnt = 0 ;
					for (int i = 0; i < A.size(); i++) {
						for (int j = 0; j < A[0].size(); j++) {
							if (A[i][j] == 1) {
								dfs(i, j, A, cnt = 0);
								mx = max(cnt, mx);
							}
						}
					}
					return mx;
				}
			}
			{
				/*
				Level Order
				https://www.interviewbit.com/problems/level-order/

				#BFS
				*/
				vector<vector<int> > Solution::levelOrder(TreeNode * A) {
					vector<vector<int>> S;
					queue<TreeNode*> Q;
					Q.push(A);
					while (!Q.empty()) {
						int n = Q.size();
						S.push_back(vector<int> ());
						while (n--) {
							TreeNode* C = Q.front();
							Q.pop();
							S.back().push_back(C->val);
							if (C->left)Q.push(C->left);
							if (C->right)Q.push(C->right);
						}
					}
					return S;
				}

			}
			{
				/*
				Smallest Multiple With 0 and 1
				https://www.interviewbit.com/problems/smallest-multiple-with-0-and-1/
				*/
				string Solution::multiple(int N) {
					if (N == 1) return "1";
					vector<int> p(N, -1); //parent state
					vector<int> s(N, -1); //step from parent to current
					//BFS
					int steps[2] = {0, 1};
					queue<int> q;
					q.push(1);
					while (!q.empty()) {
						int curr = q.front();
						q.pop();
						if (curr == 0) break;
						for (int step : steps) {
							int next = (curr * 10 + step) % N;
							if (p[next] == -1) {
								p[next] = curr;
								s[next] = step;
								q.push(next);
							}
						}
					}
					//build reversed string
					string number;
					for (int it = 0; it != 1; it = p[it])
						number.push_back('0' + s[it]);
					number.push_back('1');
					//return the reverse if it
					return string(number.rbegin(), number.rend());
				}
			}
			{
				/*
				Snake Ladder Problem!
				https://www.interviewbit.com/problems/snake-ladder-problem/

				#Famous
				*/
				int Solution::snakeLadder(vector<vector<int> > &A, vector<vector<int> > &B) {
					vector<int> Ladder(101), Snake(101);
					for (auto x : A) Ladder[x[0]] = x[1];
					for (auto x : B) Snake[x[0]] = x[1];

					vector<bool> vis(101);
					queue<int> Q;
					int step = 0;
					Q.push(1);

					while (!Q.empty()) {
						int n = Q.size();
						while (n--) {
							int position = Q.front();
							Q.pop();
							if (position == 100) return step;
							if (position > 100 || vis[position]) continue;
							vis[position] = true;
							for (int i = 1; i <= 6; i++) {
								int pos = position + i;
								if (Ladder[pos]) {
									Q.push(Ladder[pos]);
								} else if (Snake[pos]) {
									Q.push(Snake[pos]);
								} else {
									Q.push(pos);
								}
							}
						}
						step++;
					}
					return -1;
				}

			}
			{
				/*
				Min Cost Path
				https://www.interviewbit.com/problems/min-cost-path/
				*/
				unordered_map<char, int> Map{
					{'U', 0},
					{'R', 1},
					{'D', 2},
					{'L', 3}
				};
				vector<pair<int, int>> D = {{ -1, 0}, {0, 1}, {1, 0}, {0, -1}};

				int Solution::solve(int A, int B, vector<string> &E) {
					priority_queue<vector<int>> Q;
					vector<vector<int>> Vis(A, vector<int>(B, INT_MAX));
					Vis[0][0] = 0;
					Q.push({0, 0, 0});
					while (!Q.empty()) {
						auto curr = Q.top();
						int i = curr[1], j = curr[2];
						Q.pop();
						auto [di, dj] = D[Map[E[i][j]]];
						int ni = i + di, nj = j + dj;
						if (ni >= 0 && ni < A && nj >= 0 && nj < B) {
							if (Vis[ni][nj] > Vis[i][j]) {
								Vis[ni][nj] = Vis[i][j];
								Q.push({ -Vis[i][j], ni, nj});
							}
						}
						for (auto [di, dj] : D) {
							ni = i + di, nj = j + dj;
							if (ni < 0 || ni >= A || nj < 0 || nj >= B) continue;
							if (Vis[ni][nj] > Vis[i][j] + 1) {
								Vis[ni][nj] = Vis[i][j] + 1;
								Q.push({ -Vis[i][j] + 1, ni, nj});
							}
						}
					}
					return Vis[A - 1][B - 1];
				}
			}
			{
				/*
				Permutation Swaps!
				https://www.interviewbit.com/problems/permutation-swaps/

				#Good
				*/
				void dfs(int ind, vector<vector<int>> &adj, int key, vector<int> &A, vector<int> &vis) {
					if (vis[A[ind] - 1]) return;
					vis[A[ind] - 1] = key;
					for (auto i : adj[ind]) {
						dfs(i, adj, key, A, vis);
					}
				}
				int Solution::solve(vector<int> &A, vector<int> &B, vector<vector<int> > &C) {
					int n = A.size();
					vector<vector<int>> adj(n, vector<int>());
					int x = C.size();
					for (int i = 0; i < x; i++) {
						adj[C[i][0] - 1].push_back(C[i][1] - 1);
						adj[C[i][1] - 1].push_back(C[i][0] - 1);
					}
					vector<int> vis(n);
					int key = 1;
					for (int i = 0; i < n; i++) {
						if (!vis[A[i] - 1]) {
							dfs(i, adj, key, A, vis);
							key++;
						}
					}
					for (int i = 0; i < n; i++) {
						if (vis[A[i] - 1] != vis[B[i] - 1])return 0;
					}
					return 1;
				}
			}
		}
		// Graph Connectivity
		{
			{
				/*
				Commutable Islands
				https://www.interviewbit.com/problems/commutable-islands/

				#Famous #Prims #Kruskal
				*/
				//Prim
				{
					int prims(vector<vector<pair<int, int>>> &adj, vector<int> &start, const int &A) {
						vector<bool> vis(A);
						priority_queue<pair<int, int>> pq;
						int ans = 0;
						pq.push({0, start[0] - 1});
						while (!pq.empty()) {
							auto [weight, src] = pq.top();
							pq.pop();
							if (vis[src]) continue;
							vis[src] = true;
							ans -= weight;
							for (auto [dest, w2] : adj[src]) {
								if (vis[dest]) continue;
								pq.push({ -w2, dest});
							}
						}
						return ans;
					}
					int Solution::solve(int A, vector<vector<int>> &B) {
						int n = B.size();
						vector<vector<pair<int, int>>> adj(A);
						vector<int> start = B[0];
						for (auto edge : B) {
							if (start[2] < edge[2]) start = edge;
							adj[edge[0] - 1].push_back({edge[1] - 1, edge[2]});
							adj[edge[1] - 1].push_back({edge[0] - 1, edge[2]});
						}
						return prims(adj, start, A);
					}
				}
				// Kruskal
				{
					bool comp(vector<int> &A, vector<int> &B) {
						if (A[2] < B[2])return 1;
						return 0;
					}
					vector<int> parent;
					int getparent(int A) {
						while (parent[A] != A) {
							A = parent[A];
						}
						return A;
					}
					void setparent(int A, int B) {
						parent[A] = parent[B];
					}
					int Solution::solve(int A, vector<vector<int>> &B) {
						int n = B.size();
						sort(B.begin(), B.end(), comp);
						parent.resize(A);
						for (int i = 0; i < A; i++) {
							parent[i] = i;
						}
						int ans = 0;
						for (int i = 0; i < n; i++) {
							int p1 = getparent(B[i][0] - 1), p2 = getparent(B[i][1] - 1);
							if (p1 != p2) {
								ans += B[i][2];
								setparent(p1, p2);
							}
						}
						return ans;
					}
				}
			}
			{
				/*
				Possibility of finishing all courses given pre-requisites
				https://www.interviewbit.com/problems/possibility-of-finishing-all-courses-given-prerequisites/
				https://leetcode.com/problems/course-schedule

				#Famous
				*/
				//Cycle Detection
				{
					bool cycle(vector<vector<int>> &adj, int ind, vector<bool> &vis, vector<bool> &vis2) {
						vis2[ind] = 1;

						if (vis[ind])return true;
						vis[ind] = 1;
						for (int x : adj[ind]) {
							if (cycle(adj, x, vis, vis2)) return 1;
						}
						vis[ind] = 0;
						return 0;
					}
					int Solution::solve(int A, vector<int> &B, vector<int> &C) {
						vector<vector<int>> adj(A, vector<int>());
						int n = B.size();
						for (int i = 0; i < n; i++) {
							adj[C[i] - 1].push_back(B[i] - 1);
						}
						vector<bool> vis(A, false), vis2 = vis;
						for (int i = 0; i < A; i++) {
							if (!vis2[i] && cycle(adj, i, vis, vis2)) return 0;
						}
						return 1;
					}
				}
				// White Grey Balck Algo : https://leetcode.com/problems/course-schedule
				// #AlgoExample
				{
					enum class State{white, grey, black};

					class Solution {
					public:
						bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
							vector<State> state(numCourses, State::white);
							unordered_map<int, vector<int>> U;

							for (auto x : prerequisites) {
								U[x[0]].push_back(x[1]);
							}
							for (int i = 0; i < numCourses; i++) {
								if (!dfs(i, U, state)) {
									return false;
								}
							}
							return true;
						}
					private:
						bool dfs(int i, unordered_map<int, vector<int>> &U, vector<State> &state) {
							if (state[i] == State::grey) return false;
							if (state[i] == State::black) return true;
							if (U.find(i) == U.end()) return true;

							state[i] = State::grey;
							for (auto &x : U[i]) {
								if (!dfs(x, U, state)) {
									return false;
								}
							}
							state[i] = State::black;
							return true;
						}
					};
				}
			}
		}
		{
			/*
			Course Schedule II
			https://leetcode.com/problems/course-schedule-ii

			#Kahn's Algo #AlgoExample #Good
			*/
			vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
				vector<vector<int>> V(numCourses);
				vector<int> Edge(numCourses), ans;
				for (auto &x : prerequisites) {
					V[x[1]].push_back(x[0]);
					Edge[x[0]]++;
				}
				queue<int> Q;
				for (int i = 0; i < numCourses; i++) {
					if (Edge[i] == 0) {
						Q.push(i);
					}
				}
				while (!Q.empty()) {
					int top = Q.front();
					Q.pop();
					for (auto &x : V[top]) {
						if (--Edge[x] == 0) {
							Q.push(x);
						}
					}
					ans.push_back(top);
				}
				if (ans.size() == numCourses) return ans;
				return {};
			}
			{
				/*
				Cycle in Undirected Graph
				https://www.interviewbit.com/problems/cycle-in-undirected-graph/
				*/
				bool cycle(int ind, int X, vector<vector<int>> &adj, vector<bool> &vis) {
					if (vis[ind]) {
						return true;
					}
					else vis[ind] = true;

					int n = adj[ind].size();
					for (int i = 0; i < n; i++) {
						if (adj[ind][i] == X)continue;
						if (cycle(adj[ind][i], ind, adj, vis)) {
							return 1;
						}
					}
					return 0;

				}
				int Solution::solve(int A, vector<vector<int> > &B) {
					vector<bool> vis(A, false);
					vector<vector<int>> adj(A, vector<int>());
					int n = B.size();
					for (int i = 0; i < n; i++) {
						adj[B[i][0] - 1].push_back(B[i][1] - 1);
						adj[B[i][1] - 1].push_back(B[i][0] - 1);
					}
					for (int i = 0; i < A; i++) {
						if (vis[i])continue;
						if (cycle(i, -1, adj, vis))return 1;
					}
					return 0;
				}

			}
			{
				/*
				Black Shapes
				https://www.interviewbit.com/problems/black-shapes/
				*/
				int n, m;
				int shapes;
				vector<pair<int, int>> D = {{1, 0}, { -1, 0}, {0, 1}, {0, -1}};

				void dfs(vector<vector<bool>> &vis, int a, int b, vector<string> &A) {
					vis[a][b] = 1;
					for (auto [i, j] : D) {
						i += a;
						j += b;
						if (i < n && j < m && i >= 0 && j >= 0 && !vis[i][j] && A[i][j] == 'X') {
							dfs(vis, i, j, A);
						}
					}

				}
				int Solution::black(vector<string> &A) {
					n = A.size(), m = A[0].size();
					vector<vector<bool>> vis(n, vector<bool>(m, 0));
					shapes = 0;
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < m; j++) {
							if (!vis[i][j] && A[i][j] == 'X') {
								shapes++;
								dfs(vis, i, j, A);
							}
						}
					}
					return shapes;
				}
			}
		}
		// Graph Adhoc
		{
			{
				/*
				Convert Sorted List to Binary Search Tree
				https://www.interviewbit.com/problems/convert-sorted-list-to-binary-search-tree/
				*/
				TreeNode* Solution::sortedListToBST(ListNode * A) {
					if (!A)return NULL;
					if (!A->next) return new TreeNode(A->val);
					ListNode *slow = A, *fast = A, *prev;
					while (fast && fast->next) {
						prev = slow;
						slow = slow->next;
						fast = fast->next->next;
					}
					TreeNode* root = new TreeNode(slow->val);
					prev->next = nullptr;
					root->left = sortedListToBST(A);
					root->right = sortedListToBST(slow->next);

					return root;
				}
			}
		}
		// Shortest Path
		{
			{
				/*
				Knight On Chess Board
				https://www.interviewbit.com/problems/knight-on-chess-board/
				*/
				vector<pair<int, int>> Step = {{ -2, 1}, { -1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, { -1, -2}, { -2, -1}};
				int Solution::knight(int A, int B, int C, int D, int E, int F) {
					C--, D--, E--, F--;

					vector<vector<int>> vis(A, vector<int> (B, INT_MAX));
					queue<pair<int, int>> Q;

					Q.push({C, D});
					vis[C][D] = 0;

					while (!Q.empty()) {
						auto [i, j] = Q.front();
						Q.pop();
						for (auto [di, dj] : Step) {
							int ni = i + di, nj = j + dj;
							if (ni < 0 || ni >= A || nj < 0 || nj >= B) continue;
							if (vis[ni][nj] > vis[i][j] + 1) {
								vis[ni][nj] = vis[i][j] + 1;
								Q.push({ni, nj});
							}
						}
					}
					if (vis[E][F] == INT_MAX)return -1;
					return vis[E][F];
				}
			}
			{
				/*
				Useful Extra Edges
				https://www.interviewbit.com/problems/useful-extra-edges/

				#Good #VeryGood #Dijkstra #AlgoExample
				*/
				void dijikstra(vector<int> &d, vector<vector<pair<int, int>>>&adj, int start)
				{
					priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>> > pq;
					pq.push({0, start});
					d[start] = 0;
					while (!pq.empty())
					{
						int cur = pq.top().second;

						pq.pop();
						for (auto it : adj[cur])
						{
							if (d[it.first] > d[cur] + it.second)
							{
								d[it.first] = d[cur] + it.second;
								pq.push({d[it.first], it.first});
							}
						}
					}
				}
				int Solution::solve(int A, vector<vector<int>> &B, int C, int D, vector<vector<int>> &E) {

					vector<vector<pair<int, int>>>adj(A + 1);
					for (int i = 0; i < B.size(); i++)
					{
						adj[B[i][0]].push_back({B[i][1], B[i][2]});
						adj[B[i][1]].push_back({B[i][0], B[i][2]});
					}
					vector<int>ds(A + 1, 10000000);
					vector<int>de(A + 1, 10000000); // not using INT_MAX becuase the dist variable (used below) will go out of bounds
					dijikstra(ds, adj, C);
					dijikstra(de, adj, D);

					int ans = ds[D];
					for (int i = 0; i < E.size(); i++)
					{
						int dist = ds[E[i][0]] + de[E[i][1]] + E[i][2];
						int dist1 = ds[E[i][1]] + de[E[i][0]] + E[i][2];
						ans = min(ans, min(dist, dist1));
					}

					if (ans != 10000000)
						return ans;
					return -1;
				}
			}
			{
				/*
				Word Ladder I
				https://www.interviewbit.com/problems/word-ladder-i/
				https://leetcode.com/problems/word-ladder/

				#Famous
				*/
				int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
					unordered_set<string> U(wordList.begin(), wordList.end());
					int steps = 1, n = wordList[0].size();
					queue<string> Q;
					Q.push(beginWord);

					while (!Q.empty()) {
						int m = Q.size();
						for (int k = 0; k < m; k++) {
							string beginWord = Q.front();
							Q.pop();
							for (int i = 0; i < n; i++) {
								char c = beginWord[i];
								for (int j = 0; j < 26; j++) {
									beginWord[i] = 'a' + j;
									if (U.find(beginWord) != U.end()) {
										U.erase(beginWord);
										if (beginWord == endWord) {
											return ++steps;
										}
										Q.push(beginWord);
									}
								}
								beginWord[i] = c;
							}
						}
						steps++;
					}
					return 0;
				}
			}
			{
				/*
				Word Ladder II
				https://www.interviewbit.com/problems/word-ladder-ii/
				https://leetcode.com/problems/word-ladder-ii/
				*/
				unordered_map<string, int> M; //map of indices
				vector<vector<int>> Final;
				int N, minpath;

				vector<vector<string>> ans(vector<string> &dict) { // for gettting final ans
					int n = Final.size();
					vector<vector<string>> S;
					for (int i = 0; i < n; i++) {
						if (Final[i].size() > minpath)continue;
						vector<string> X;
						for (int x : Final[i]) {
							X.push_back(dict[x]);
						}
						S.push_back(X);
					}
					return S;
				}
				void change(int ind, vector<int> &T, vector<string> &dict, vector<bool> &vis) {
					int t = T.size();
					if (minpath <= t)return;
					T.push_back(ind);
					vis[ind] = 1;
					if (ind == N - 1) {
						minpath = min(minpath, t + 1);
						Final.push_back(T);
					}
					else {
						string z = dict[ind];
						int l = z.length();
						for (int i = 0; i < l; i++) {
							string x = z;
							for (int j = 0; j < 26; j++) {
								x[i] = 'a' + j;
								if (M.find(x) != M.end() && vis[M[x]] == 0) {
									change(M[x], T, dict, vis);
								}
							}
						}
					}
					vis[ind] = 0;
					T.pop_back();
				}
				vector<vector<string> > Solution::findLadders(string start, string end, vector<string> &dict) {
					if (start == end)return {{start}};
					M.clear();
					Final.clear();
					dict.push_back(start);
					dict.push_back(end);
					N = dict.size();
					minpath = N;

					for (int i = 0; i < N; i++) {
						M[dict[i]] = i;
					}
					vector<int> T; //sequence in form of indices
					vector<bool> vis(N, 0);

					change(N - 2, T, dict, vis);
					return ans(dict);
				}
			}

		}
		// Graph Hashing
		{
			{
				/*
				Clone Graph
				https://www.interviewbit.com/problems/clone-graph/
				*/

				UndirectedGraphNode * clone(unordered_map<UndirectedGraphNode *, UndirectedGraphNode *> &M, UndirectedGraphNode * node) {
					if (M.find(node) != M.end())return M[node];
					UndirectedGraphNode *A = new UndirectedGraphNode(node->label);
					M[node] = A;
					for (auto &x : node->neighbors) {
						A->neighbors.push_back(clone(M, x));
					}
					return A;
				}
				UndirectedGraphNode *Solution::cloneGraph(UndirectedGraphNode * node) {
					unordered_map<UndirectedGraphNode *, UndirectedGraphNode *> M;
					return clone(M, node);
				}
			}
		}
	}
// DP
	{
		// 2D string DP
		{
			{
				/*
				Longest Common Subsequence
				https://www.interviewbit.com/problems/longest-common-subsequence/

				#Famous #LongestCommonSunsequence #LCS #AlgoExample #BottomUp #Tabulation
				*/
				int Solution::solve(string A, string B) {
					int n = A.length(), m = B.length();
					vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= m; j++) {
							if (A[i - 1] == B[j - 1]) {
								dp[i][j] = 1 + dp[i - 1][j - 1];
							} else {
								dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
							}
						}
					}
					return dp[n][m];
				}
			}
			{
				/*
				Longest Palindromic Subsequence
				https://interviewbit.com/problems/longest-palindromic-subsequence/

				#Famous #TopDown #AlgoExample #Memoisation
				*/
				int pal(string & A, int s, int e, vector<vector<int>> &dp) {
					if (dp[s][e] != -1) return dp[s][e];
					int ans;
					if (s + 1 == e) {
						if (A[s] == A[e]) {
							ans = 2;
						} else {
							ans = 1;
						}
					}
					else if (A[s] == A[e]) {
						ans = 2 + pal(A, s + 1, e - 1, dp);
					}
					else {
						ans = max({pal(A, s + 1, e, dp), pal(A, s, e - 1, dp)});
					}
					dp[s][e] = ans;
					return ans;
				}
				int Solution::solve(string A) {
					int n = A.length();
					vector<vector<int>> dp(n, vector<int> (n, -1));
					for (int i = 0; i < n; i++)dp[i][i] = 1;
					return pal(A, 0, n - 1, dp);
				}
			}
			{
				/*
				Edit Distance
				https://www.interviewbit.com/problems/edit-distance/

				#Famous
				*/
				int Solution::minDistance(string A, string B) {
					int n = A.length(), m = B.length();
					vector<vector<int>> dp(n + 1, vector<int> (m + 1, 0));
					for (int j = 0; j <= m; j++) {
						dp[0][j] = j;
					}
					for (int i = 0; i <= n; i++) {
						dp[i][0] = i;
					}
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= m; j++) {
							if (A[i - 1] == B[j - 1]) {
								dp[i][j] = dp[i - 1][j - 1];
							} else {
								dp[i][j] = min({dp[i][j - 1], //delete char from A
								                dp[i - 1][j], //insert a char in A
								                dp[i - 1][j - 1]
								               }//replace last character in A
								              ) + 1 ;
							}
						}
					}
					return dp[n][m];
				}
			}
			{
				/*
					Repeating Sub-Sequence
					https://www.interviewbit.com/problems/repeating-subsequence/
				*/
				int Solution::anytwo(string A) {
					int n = A.length();
					vector<vector<int>> M(n + 1, vector<int>(n + 1, 0));
					for (int i = 1; i < n + 1; i++) {
						for (int j = 1; j < n + 1; j++) {
							if (A[i - 1] == A[j - 1] && i != j) {
								M[i][j] = M[i - 1][j - 1] + 1;
							} else {
								M[i][j] = max(M[i - 1][j], M[i][j - 1]);
							}
						}
					}
					if (M[n][n] > 1)return 1;
					return 0;
				}
			}
			{
				/*
				Distinct Subsequences
				https://www.interviewbit.com/problems/distinct-subsequences/
				*/
				int Solution::numDistinct(string A, string B) {
					int n = A.length(), m = B.length();
					vector<vector<int>> dp(n + 1, vector<int> (m + 1, 0));
					for (int i = 0; i <= n; i++) {
						dp[i][0] = 1;
					}
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= m; j++) {
							if (A[i - 1] != B[j - 1]) {
								dp[i][j] = dp[i - 1][j];
							} else {
								dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
							}
						}
					}
					return dp[n][m];
				}
			}
			{
				/*
				Scramble String
				https://www.interviewbit.com/problems/scramble-string/
				*/
				unordered_map<string, unordered_map<string, bool>> M;
				bool helper(string A, string B) {
					if (M.find(A) != M.end() && M[A].find(B) != M[A].end()) {
						return M[A][B];
					}
					if (A == B) {
						M[A][B] = 1;
						return 1;
					}
					int n = A.length();
					if (n == 1)return 0;

					for (int i = 1; i < n; i++) {
						if ((helper(A.substr(0, i), B.substr(n - i)) && helper(A.substr(i), B.substr(0, n - i)))
						        || (helper(A.substr(0, i), B.substr(0, i)) && helper(A.substr(i), B.substr(i)))) {
							M[A][B] = 1;
							return 1;
						}
					}
					M[A][B] = 0;
					return 0;
				}
				int Solution::isScramble(const string A, const string B) {
					if (A.length() != B.length())return 0;
					if (A.length() == 0)return 1;
					M.clear();
					return helper(A, B);
				}
			}
			{
				/*
				Regular Expression Match
				https://www.interviewbit.com/problems/regular-expression-match/

				#Good
				*/
				int Solution::isMatch(const string A, const string B) {
					int n = A.length(), m = B.length();
					vector<vector<bool>> M(n + 1, vector<bool> (m + 1, 0));
					M[0][0] = true;
					for (int i = 1; i <= m; i++) {
						if (B[i - 1] == '*') M[0][i] = 1;
						break;
					}
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= m; j++) {
							if (A[i - 1] == B[j - 1] || B[j - 1] == '?') {
								M[i][j] = M[i - 1][j - 1];
							} else if (B[j - 1] == '*') {
								M[i][j] = M[i - 1][j] || M[i][j - 1];
							}
						}
					}
					return M[n][m];
				}
			}
			{
				/*
				Regular Expression II
				https://www.interviewbit.com/problems/regular-expression-ii/

				#Good
				*/
				bool comp(char a, char b) {
					if (a == b) return 1;
					else if (b == '.') return 1;
					return 0;
				}
				int Solution::isMatch(const string A, const string B) {
					int n = A.length(), m = B.length();
					vector<vector<bool>> dp(n + 1, vector<bool>(m + 1, 0));
					dp[0][0] = 1;
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= m; j++) {
							if (comp(A[i - 1], B[j - 1])) {
								dp[i][j] = dp[i - 1][j - 1];
							} else if (B[j - 1] == '*') {
								dp[i][j] = dp[i][j - 2] // 0 repeat
								           || dp[i][j - 1]; // 1 repeat
								if (comp(A[i - 1], B[j - 2])) {
									dp[i][j] = dp[i][j] || dp[i - 1][j]; // more repeat
								}
							}
						}
					}
					return dp[n][m];
				}

			}
			{
				/*
				Interleaving Strings
				https://www.interviewbit.com/problems/interleaving-strings/
				*/
				int Solution::isInterleave(string A, string B, string C) {
					int a = A.length(), b = B.length(), c = C.length();
					vector<vector<int>> dp(a + 1, vector<int> (b + 1, 0));
					dp[0][0] = 1;
					for (int i = 1; i <= a && A[i - 1] == C[i - 1]; i++) dp[i][0] = 1;
					for (int j = 1; j <= b && B[j - 1] == C[j - 1]; j++) dp[0][j] = 1;

					for (int i = 1; i <= a; i++) {
						for (int j = 1; j <= b; j++) {
							if ((dp[i - 1][j] && A[i - 1] == C[i + j - 1]) || (dp[i][j - 1] && B[j - 1] == C[i + j - 1])) {
								dp[i][j] = 1;
							}
						}
					}
					return dp[a][b];
				}
			}
		}
		// Simple array DP
		{
			{
				/*
				Length of Longest Subsequence
				https://www.interviewbit.com/problems/length-of-longest-subsequence/

				#Famous
				*/
				int Solution::longestSubsequenceLength(const vector<int> &A) {
					vector<int> G(A.size(), 1), S = G;
					for (int i = 0; i < A.size(); i++) {
						for (int j = 0; j < i; j++) {
							if (A[i] > A[j] && G[i] < G[j] + 1) {
								G[i] = G[j] + 1;
							}
						}
					}
					for (int i = A.size() - 1; i >= 0; i--) {
						for (int j = A.size() - 1; j > i; j--) {
							if (A[i] > A[j] && S[i] < S[j] + 1) {
								S[i] = S[j] + 1;
							}
						}
					}
					int ans = 0;
					for (int i = 0; i < A.size(); i++) {
						ans = max(ans, G[i] + S[i] - 1);
					}
					return ans;
				}
			}
			{
				/*
				Smallest sequence with given Primes
				https://www.interviewbit.com/problems/smallest-sequence-with-given-primes/
				*/
				vector<int> Solution::solve(int A, int B, int C, int D) {
					set<int> s;
					s.insert(A);
					s.insert(B);
					s.insert(C);
					vector<int> arr;
					if (D == 0)
						return arr;
					while (!s.empty())
					{
						int a = *(s.begin());
						s.erase(s.begin());
						arr.push_back(a);
						if (arr.size() == D)
							break;
						s.insert(a * A);
						s.insert(a * B);
						s.insert(a * C);
					}
					return arr;
				}
			}
			{
				/*
				Largest area of rectangle with permutations
				https://www.interviewbit.com/problems/largest-area-of-rectangle-with-permutations/

				#Good #VeryGood #Tricky
				*/
				int area(vector<int> M) {
					sort(M.begin(), M.end());
					int ans = 0;
					for (int i = 0; i < M.size(); i++) {
						ans = max(ans, M[i] * (M.size() - i));
					}
					return ans;
				}
				int Solution::solve(vector<vector<int>> &A) {
					int n = A.size(), m = A[0].size();
					vector<vector<int>> M(n, vector<int> (m, 0));
					for (int i = n - 1, i >= 0; i--) {
						for (int j = 0; j < m; j++) {
							if (A[i][j] == 1) {
								if (i == n - 1) {
									M[i][j] = 1;
								} else {
									M[i][j] = M[i + 1][j] + 1;
								}
							}
						}
					}
					int ans = 0;
					for (int i = 0; i < n; i++) {
						ans = max(ans, area(M[i]));
					}
					return ans;
				}
			}
			{
				/*
				Tiling With Dominoes
				https://www.interviewbit.com/problems/tiling-with-dominoes/

				#Good
				*/
				long long M 1000000007
				int Solution::solve(int n) {
					if (n % 2 != 0)return 0;
					vector<long long> A(n + 1, 0), B(n + 1, 0);
					A[0] = 1;
					B[1] = 1;
					for (int i = 2; i <= n; i += 2) {
						A[i] = (A[i - 2] + 2 * B[i - 1]) % M;
						B[i + 1] = (A[i] + B[i - 1]) % M;
					}
					return A[n];
				}

			}
			{
				/*
				Paint House!
				https://www.interviewbit.com/problems/paint-house/
				*/
				int Solution::solve(vector<vector<int> > &A) {
					vector<vector<int>> C(A.size(), vector<int>(3, -1));
					int n = A.size();
					C[0] = A[0];
					for (int i = 1; i < n; i++) {
						for (int j = 0; j < 3; j++) {
							C[i][j] = min(C[i - 1][(j + 1) % 3], C[i - 1][(j + 2) % 3]) + A[i][j];
						}
					}
					return min({C[n - 1][0], C[n - 1][1], C[n - 1][2]});
				}
			}
			{
				/*
				Ways to Decode
				https://www.interviewbit.com/problems/ways-to-decode/
				*/
				int Solution::numDecodings(string A) {
					int N = A.length();
					int dp[N + 1] = {};
					dp[0] = 1;
					dp[1] = A[0] == '0' ? 0 : 1;
					for (int i = 2; i <= N; i++) {
						int oneDigit = stoi(A.substr(i - 1, 1));
						int twoDigit = stoi(A.substr(i - 2, 2));
						if (oneDigit >= 1) {
							dp[i] += dp[i - 1];
						}
						if (twoDigit <= 26) {
							dp[i] += dp[i - 2];
						}
						dp[i] %= 1000000007;
					}
					return dp[N];
				}
			}
			{
				/*
				Stairs
				https://www.interviewbit.com/problems/stairs/

				#Famous
				*/
				int Solution::climbStairs(int A) {
					vector<int> dp(A + 1, 0);
					dp[0] = 1;
					dp[1] = 1;
					for (int i = 2; i <= A; i++) {
						dp[i] = dp[i - 1] + dp[i - 2];
					}
					return dp[A];
				}
			}
			{
				/*
				Longest Increasing Subsequence
				https://www.interviewbit.com/problems/longest-increasing-subsequence/

				#Famous
				*/
				int Solution::lis(const vector<int> &A) {
					int n = A.size();
					if (n == 0)return 0;
					vector<int> lcs(n, 1);
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < i; j++) {
							if (A[i] > A[j] && lcs[i] < 1 + lcs[j]) {
								lcs[i] = 1 + lcs[j];
							}
						}
					}
					return *max_element(lcs.begin(), lcs.end());
				}
			}
			{
				/*
				Intersecting Chords in a Circle
				https://www.interviewbit.com/problems/intersecting-chords-in-a-circle/
				*/
				long long M 1000000007;
				int helper(int A, vector<long long int> &dp) {
					if (dp[A] != -1) return dp[A];
					long long int x = 0;
					for (int i = 0; i < A; i++) {
						x += helper(i, dp) * helper(A - i - 1, dp);
						x %= M;
					}
					dp[A] = x;
					return x;
				}
				int Solution::chordCnt(int A) {
					vector<long long int> dp(A + 1, -1);
					if (A == 0)return 0;
					if (A == 1)return 1;
					dp[0] = 1;
					dp[1] = 1;
					dp[2] = 2;
					return helper(A, dp);
				}

			}
		}
		// Greedy OR DP
		{
			{
				/*
				Tushar's Birthday Bombs
				https://www.interviewbit.com/problems/tushars-birthday-bombs/

				#Good
				*/
				vector<int> Solution::solve(int A, vector<int> &B) {
					int n = A, m = B.size();
					vector<int> dp(A + 1, -1), back(A + 1, 0), ans;
					for (int i = 1; i <= A; i++) {
						for (int j = 0; j < m; j++) {
							if (i >= B[j] && dp[i] < 1 + dp[i - B[j]]) {
								dp[i] = 1 + dp[i - B[j]];
								back[i] = j;
							}
						}
					}
					while (A > 0 && A - B[back[A]] >= 0) {
						ans.push_back(back[A]);
						A -= B[back[A]];
					}
					return ans;
				}
			}
			{
				/*
				Jump Game Array
				https://www.interviewbit.com/problems/jump-game-array/
				*/
				int Solution::canJump(vector<int> &A) {
					int n = A.size(), maxi = 0;
					for (int i = 0; i < n; i++) {
						if (maxi < i)return 0;
						maxi = max(maxi, i + A[i]);
					}
					return 1;
				}
			}
			{
				/*
				Min Jumps Array
				https://www.interviewbit.com/problems/min-jumps-array/

				#Famous
				*/
				int Solution::jump(vector<int> &A) {
					if (A.size() <= 1) return 0;
					int maxi = A[0], ans = 1;
					for (int i = 1; i < A.size(); i++) {
						if (maxi >= A.size() - 1) return ans;
						if (maxi < i) return -1;

						int cm = maxi;
						for (int j = i; j <= cm; j++) {
							maxi = max(maxi, j + A[j]);
						}
						i = cm;
						ans++;
					}
					return ans;
				}
			}

		}
		// DP tricky
		{
			{
				/*
				N digit numbers with digit sum S
				https://www.interviewbit.com/problems/n-digit-numbers-with-digit-sum-s-/
				*/
				long long M 1000000007;
				int Solution::solve(int A, int B) {
					if (B == 0 || A == 0)return 0;

					vector<vector<int>> dp(A + 1, vector<int> (B + 1, 0));
					for (int j = 1; j <= min(9, B); j++) {
						dp[1][j] = 1;
					}

					for (int i = 2; i <= A; i++) {
						for (int j = 1; j <= B; j++) {
							for (int k = 0; k <= 9 && j - k >= 0; k++) {
								dp[i][j] += dp[i - 1][j - k];
								dp[i][j] %= M;
							}
						}
					}
					return dp[A][B];
				}
			}
			{
				/*
				Ways to color a 3xN Board
				https://www.interviewbit.com/problems/ways-to-color-a-3xn-board/
				*/
				int Solution::solve(int A) {
					int dp[4][4][4][A + 1];
					vector<vector<int>> valid;
					for (int i = 0; i < 4; i++) {
						for (int j = 0; j < 4; j++) {
							if (j == i)continue;
							for (int k = 0; k < 4; k++) {
								if (j == k)continue;
								dp[i][j][k][1] = 1;
								valid.push_back({i, j, k});
							}
						}
					}
					for (int i = 2; i <= A; i++) {
						for (auto &x : valid) {
							dp[x[0]][x[1]][x[2]][i] = 0;
							for (auto &y : valid) {
								if (x[0] == y[0] || x[1] == y[1] || x[2] == y[2])continue;
								dp[x[0]][x[1]][x[2]][i] += dp[y[0]][y[1]][y[2]][i - 1];
								dp[x[0]][x[1]][x[2]][i] %= 1000000007;
							}
						}
					}
					int ans = 0;
					for (auto &x : valid) {
						ans += dp[x[0]][x[1]][x[2]][A];
						ans %= 1000000007;
					}
					return ans;
				}
			}
			{
				/*
				Kth Manhattan Distance Neighbourhood
				https://www.interviewbit.com/problems/kth-manhattan-distance-neighbourhood/
				*/
				vector<vector<int> > Solution::solve(int A, vector<vector<int> > &B) {
					int n = B.size(), m = B[0].size();
					vector<vector<int>> S(n, vector<int>(m, 0));
					int M[n][m][A + 1];
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < m; j++) {
							for (int k = 0; k <= A; k++) {
								M[i][j][k] = max({
									(j - k >= 0) ? B[i][j - k] : INT_MIN,
									(j + k < m) ? B[i][j + k] : INT_MIN,
									(k != 0) ? M[i][j][k - 1] : INT_MIN
								});
							}
						}
					}
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < m; j++) {
							for (int k = 0; k <= A; k++) {
								S[i][j] = max({ S[i][j],
								                (i - k >= 0) ? M[i - k][j][A - k] : INT_MIN,
								                (i + k < n) ? M[i + k][j][A - k] : INT_MIN
								              });
							}
						}
					}
					return S;
				}
			}
			{
				/*
				Best Time to Buy and Sell Stocks I
				https://www.interviewbit.com/problems/best-time-to-buy-and-sell-stocks-i/

				#Famous
				*/
				int Solution::maxProfit(const vector<int> &A) {
					int n = A.size();
					if (n <= 1) return 0;
					int ans = 0, mini = A[0];
					for (int i = 1; i < n; i++) {
						ans = max(ans, A[i] - mini);
						mini = min(mini, A[i]);
					}
					return ans;
				}
			}
			{
				/*
				Best Time to Buy and Sell Stocks II
				https://www.interviewbit.com/problems/best-time-to-buy-and-sell-stocks-ii/

				#Famous
				*/
				int Solution::maxProfit(const vector<int> &A) {
					int n = A.size();
					int ans = 0;
					for (int i = 1; i < n; i++) {
						if (A[i] > A[i - 1]) ans += A[i] - A[i - 1];
					}
					return ans;
				}
			}
			{
				/*
				Best Time to Buy and Sell Stock atmost B times
				https://www.interviewbit.com/problems/best-time-to-buy-and-sell-stock-atmost-b-times/

				#Famous #Revise #GoodSolution
				*/
				int Solution::solve(vector<int> &A, int B) {
					int n = A.size();
					B = min(B, n / 2);
					vector<vector<int>> M(B + 1, vector<int> (n + 1));
					for (int i = 1; i <= B; i++) {
						int diff = INT_MIN;
						for (int j = 1; j <= A.size(); j++) {
							diff = max(diff, M[i - 1][j] - A[j - 1]);
							M[i][j] = max(M[i][j - 1], A[j - 1] + diff);
						}
					}
					return M[B][A.size()];
				}
			}
			{
				/*
				Coins in a Line
				https://www.interviewbit.com/problems/coins-in-a-line/

				#Good
				*/
				vector<int> sum;
				vector<vector<int>> dp;
				int pick(vector<int> &A, int s, int e) {
					if (dp[s][e] != -1)return dp[s][e];
					if (e - s <= 1) {
						dp[s][e] = max(A[s], A[e]);
						return dp[s][e];
					}
					dp[s][e] = sum[e + 1] - sum[s] - min(pick(A, s + 1, e), pick(A, s, e - 1));
					return dp[s][e];
				}
				int Solution::maxcoin(vector<int> &A) {
					int n = A.size();
					dp = vector<vector<int>> (n, vector<int>(n, -1));
					sum = vector<int>(n + 1);
					for (int i = 1; i <= n; i++) {
						sum[i] = sum[i - 1] + A[i - 1];
					}
					return pick(A, 0, n - 1);
				}
			}
			{
				/*
				Evaluate Expression To True
				https://www.interviewbit.com/problems/evaluate-expression-to-true/
				*/
				vector<vector<vector<int>>> M;
				int count(string & A, int s, int e, int state) {
					if (M[s][e][state] != -1) {
						return M[s][e][state];
					}
					int ansT = 0, ansF = 0;
					for (int i = s; i < e; i++) {
						char opr = A[2 * i + 1];
						if (opr == '|') {
							ansT += count(A, s, i, 1) * count(A, i + 1, e, 1);
							ansT += count(A, s, i, 0) * count(A, i + 1, e, 1);
							ansT += count(A, s, i, 1) * count(A, i + 1, e, 0);
							ansF += count(A, s, i, 0) * count(A, i + 1, e, 0);
						} else if (opr == '&') {
							ansT += count(A, s, i, 1) * count(A, i + 1, e, 1);
							ansF += count(A, s, i, 0) * count(A, i + 1, e, 1);
							ansF += count(A, s, i, 1) * count(A, i + 1, e, 0);
							ansF += count(A, s, i, 0) * count(A, i + 1, e, 0);
						} else if (opr == '^') {
							ansF += count(A, s, i, 1) * count(A, i + 1, e, 1);
							ansT += count(A, s, i, 0) * count(A, i + 1, e, 1);
							ansT += count(A, s, i, 1) * count(A, i + 1, e, 0);
							ansF += count(A, s, i, 0) * count(A, i + 1, e, 0);
						}
						ansT %= 1003, ansF %= 1003;
					}
					M[s][e][0] = ansF;
					M[s][e][1] = ansT;
					return M[s][e][state];
				}
				int Solution::cnttrue(string A) {
					int n = (A.length() + 1) / 2;
					M = vector<vector<vector<int>>> (n, vector<vector<int>> (n, vector<int>(2, -1)));

					for (int i = 0; i < n; i++) {
						bool state = A[2 * i] == 'T';
						M[i][i][state] = 1;
						M[i][i][!state] = 0;
					}
					return count(A, 0, n - 1, 1);
				}
			}
			{
				/*
				Egg Drop Problem!
				https://www.interviewbit.com/problems/egg-drop-problem/

				#Famous
				*/
				int Solution::solve(int eggs, int floors)
				{
					vector<vector<int>> dp(floors + 1, vector<int> (eggs + 1, 0));
					int moves = 0;
					while (dp[moves][eggs] < floors) {
						moves++;
						for (int egg = 1; egg <= eggs; egg++) {
							dp[moves][egg] = dp[moves - 1][egg] + 1 + dp[moves - 1][egg - 1];
							// not breaking + 1 + breakage
						}
					}
					return moves;
				}
			}
			{
				/*
				Longest valid Parentheses
				https://www.interviewbit.com/problems/longest-valid-parentheses/

				#Good
				*/
				int Solution::longestValidParentheses(string A) {
					int n = A.length();
					vector<int> S = { -1};
					for (int i = 0; i < n; i++) {
						if (S.back() != -1 && A[S.back()] == '(' && A[i] == ')') {
							S.pop_back();
						} else {
							S.push_back(i);
						}
					}
					S.push_back(n);
					int ans = 0;
					int n2 = S.size();
					for (int i = 1; i < n2; i++) {
						ans = max(ans, S[i] - S[i - 1] - 1);
					}
					return ans;
				}
			}

		}
		// tree DP
		{
			{
				/*
				Max edge queries!
				https://www.interviewbit.com/problems/max-edge-queries/
				*/
				{
					vector<int> level, parent;
					unordered_map<int, unordered_map<int, int>> dp, &L = dp;
					int findlevel(int i) {
						if (level[i] != -1)return level[i];
						return level[i] = 1 + findlevel(parent[i]);
					}
					int lca(int a, int b) {
						if (a < b)swap(a, b);
						if (a == b) {
							return a;
						}
						if (L.find(a) != L.end() && L[a].find(b) != L[a].end())return L[a][b];
						else if (findlevel(a) == findlevel(b)) {
							L[a][b] = lca(parent[a], parent[b]);
						}
						else if (findlevel(a) < findlevel(b)) {
							L[a][b] = lca(a, parent[b]);
						}
						else if (findlevel(a) > findlevel(b)) {
							L[a][b] = lca(parent[a], b);
						}
						return L[a][b];
					}

					int helper(int s, int e);
					int helper2(int a, int c) {
						int s = min(a, c), e = max(a, c);
						if (dp.find(s) != dp.end() && dp[s].find(e) != dp[s].end())return dp[s][e];
						return dp[s][e] = max(helper2(a, parent[c]), helper(c, parent[c]));
					}
					int helper(int s, int e) {
						if (s > e)swap(s, e);
						if (dp.find(s) != dp.end() && dp[s].find(e) != dp[s].end())return dp[s][e];
						int b = lca(s, e);
						if (b == s) {
							return dp[s][e] = helper2(s, e);
						}
						if (b == e) {
							return dp[s][e] = helper2(e, s);
						}
						return dp[s][e] = max(helper(s, b), helper(b, e));
					}

					void gclear(int N) {
						dp.clear();
						parent = vector<int> (N, -1);
						level = parent;

						parent[0] = 0;
						level[0] = 0;
					}

					vector<int> Solution::solve(vector<vector<int> > &A, vector<vector<int> > &B) {
						int N = A.size() + 1, Q = B.size();
						gclear(N);

						for (int i = 0; i < N - 1; i++) {
							if (A[i][0] > A[i][1]) swap(A[i][0], A[i][1]);
							dp[A[i][0] - 1][A[i][1] - 1] = A[i][2];
							dp[i][i] = 0;
						}
						sort(A.begin(), A.end());
						for (int i = 0; i < N - 1; i++) {
							if (parent[A[i][0] - 1] == -1) {
								parent[A[i][0] - 1] = A[i][1] - 1;
							}
							else if (parent[A[i][1] - 1] == -1) {
								parent[A[i][1] - 1] = A[i][0] - 1;
							}
						}
						vector<int> S;
						for (int i = 0; i < Q; i++) {
							if (B[i][0] > B[i][1]) swap(B[i][0], B[i][1]);
							S.push_back(helper(B[i][0] - 1, B[i][1] - 1));
						}
						return S;
					}
				}
			}
			{
				/*
				Max Sum Path in Binary Tree
				https://www.interviewbit.com/problems/max-sum-path-in-binary-tree/
				*/
				int mx;
				int onepath(TreeNode * A) {
					if (!A)return 0;
					int left = max(0, onepath(A->left));
					int right = max(0, onepath(A->right));
					mx = max(mx, left + right + A->val);
					return A->val + max(left, right);
				}
				int Solution::maxPathSum(TreeNode * A) {
					mx = INT_MIN;
					onepath(A);
					return mx;
				}
			}
		}
		// Matrix DP
		{
			{
				/*
				Kingdom War
				https://www.interviewbit.com/problems/kingdom-war/
				*/
				int Solution::solve(vector<vector<int> > &A) {
					int n = A.size(), m = A[0].size(), ans = INT_MIN;
					A.push_back(vector<int>(m + 1));
					for (int i = n - 1; i >= 0; i--) {
						A[i].push_back(0);
						for (int j = m - 1; j >= 0; j--) {
							A[i][j] += A[i + 1][j] + A[i][j + 1] - A[i + 1][j + 1];
							ans = max(ans, A[i][j]);
						}
					}
					return ans;
				}
			}
			{
				/*
				Maximum Path in Triangle
				https://www.interviewbit.com/problems/maximum-path-in-triangle/
				*/
				int Solution::solve(vector<vector<int> > &A) {
					int n = A.size();
					vector<vector<int>> dp(n, vector<int> (n, 0));
					dp[0][0] = A[0][0];
					for (int i = 1; i < n; i++) {
						dp[i][0] = dp[i - 1][0] + A[i][0];
						for (int j = 1; j <= i; j++) {
							dp[i][j] = A[i][j] + max(dp[i - 1][j], dp[i - 1][j - 1]);
						}
					}
					return *max_element(dp[n - 1].begin(), dp[n - 1].end());
				}
			}
			{
				/*
				Maximum Size Square Sub-matrix
				https://www.interviewbit.com/problems/maximum-size-square-sub-matrix/

				#Famous
				*/
				int Solution::solve(vector<vector<int> > &A) {
					int n = A.size(), m = A[0].size(), S = 0;
					vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= m; j++) {
							if (A[i - 1][j - 1] == 1) {
								dp[i][j] = 1 + min({dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]});
								S = max(dp[i][j], S);
							}
						}
					}
					return S * S;
				}
			}
			{
				/*
				Increasing Path in Matrix
				https://www.interviewbit.com/problems/increasing-path-in-matrix/
				*/
				int Solution::solve(vector<vector<int> > &A) {
					if (A.size() == 0 || A[0].size() == 0)return -1;
					vector<vector<int>> M(A.size(), vector<int>(A[0].size(), -1));
					M[0][0] = 1;
					for (int i = 0; i < A.size(); i++) {
						for (int j = 0; j < A[0].size(); j++) {
							if (M[i][j] == -1)continue;
							if (i + 1 < A.size() && A[i + 1][j] > A[i][j]) {
								M[i + 1][j] = max(M[i + 1][j], M[i][j] + 1);
							}
							if (j + 1 < A[0].size() && A[i][j + 1] > A[i][j]) {
								if (M[i][j + 1] > M[i][j] + 1)break;
								M[i][j + 1] = M[i][j] + 1;
							}
						}
					}
					return M[A.size() - 1][A[0].size() - 1];
				}
			}
			{
				/*
				Minimum Difference Subsets!
				https://www.interviewbit.com/problems/minimum-difference-subsets/
				*/
			}
			{
				/*
				Subset Sum Problem!
				https://www.interviewbit.com/problems/subset-sum-problem/
				*/
				{
					int Solution::solve(vector<int> &A, int B) {
						int n = A.size();
						vector<vector<bool>> V(n + 1, vector<bool>(B + 1));
						V[0][0] = true;
						for (int i = 1; i <= n; i++) {
							V[i][0] = true;
							for (int j = 1; j <= B; j++) {
								V[i][j] = V[i - 1][j];
								if (A[i - 1] > j) continue;
								V[i][j] = V[i][j] || V[i - 1][j - A[i - 1]];
							}
						}
						return V[n][B];
					}
				}
				{
					int Solution::solve(vector<int> &A, int B) {
						int n = A.size();
						set<int> M, D;
						M.insert(0);
						for (int i = 0; i < n; i++) {
							for (auto x : M) {
								D.insert(x);
								if (x + A[i] == B)return 1;
								if (x + A[i] > B)break;
								D.insert(x + A[i]);
							}
							M = D;
							D.clear();
						}
						return 0;
					}
				}
			}
			{
				/*
				Unique Paths in a Grid
				https://www.interviewbit.com/problems/unique-paths-in-a-grid/
				*/
				int Solution::solve(vector<int> &A, int B) {
					int n = A.size();
					vector<vector<bool>> V(n + 1, vector<bool>(B + 1));
					V[0][0] = true;
					for (int i = 1; i <= n; i++) {
						V[i][0] = true;
						for (int j = 1; j <= B; j++) {
							V[i][j] = V[i - 1][j];
							if (A[i - 1] > j) continue;
							V[i][j] = V[i][j] || V[i - 1][j - A[i - 1]];
						}
					}
					return V[n][B];
				}
			}
			{
				/*
				Dungeon Princess
				https://www.interviewbit.com/problems/dungeon-princess/

				#Revise #Good
				*/
				int Solution::calculateMinimumHP(vector<vector<int> > &A) {
					int m = A.size(), i, j;
					if (m == 0) {
						return 0;
					}
					int n = A[0].size();
					if (n == 0) {
						return 0;
					}
					int dp[m][n];
					dp[m - 1][n - 1] = std::max(0 - A[m - 1][n - 1], 0);
					for (i = m - 2; i >= 0; --i) {
						dp[i][n - 1] = std::max(dp[i + 1][n - 1] - A[i][n - 1], 0);
					}
					for (j = n - 2; j >= 0; --j) {
						dp[m - 1][j] = std::max(dp[m - 1][j + 1] - A[m - 1][j], 0);
					}

					for (i = m - 2; i >= 0; --i) {
						for (j = n - 2; j >= 0; --j) {
							dp[i][j] = std::max(std::min(dp[i][j + 1], dp[i + 1][j]) - A[i][j], 0);
						}
					}

					return dp[0][0] + 1;
				}
			}
			{
				/*
				Min Sum Path in Matrix
				https://www.interviewbit.com/problems/min-sum-path-in-matrix/
				*/
				int Solution::minPathSum(vector<vector<int> > &A) {
					int n = A.size(), m = A[0].size();
					vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= m; j++) {
							if (i == 1) {
								dp[i][j] = dp[i][j - 1] + A[0][j - 1];
								continue;
							}
							if (j == 1) {
								dp[i][j] = dp[i - 1][j] + A[i - 1][0];
								continue;
							}
							dp[i][j] = A[i - 1][j - 1] + min(dp[i - 1][j], dp[i][j - 1]);
						}
					}
					return dp[n][m];
				}
			}
			{
				/*
				Rod Cutting
				https://www.interviewbit.com/problems/rod-cutting/

				#Good #Famous
				*/
				{
					// Recurssive
					vector<vector<pair<int, int>>> DP;
					void fill(vector<int> &V, int s, int e, vector<int> &B) {
						auto cut = DP[s][e];
						if (cut.second != INT_MAX) {
							V.push_back(B[cut.second]);
							fill(V, s, cut.second, B);
							fill(V, cut.second, e, B);
						}
					}
					pair<int, int> cutRod(vector<int> &B, int sCut, int eCut) {
						if (sCut + 1 == eCut) {
							return DP[sCut][eCut] = {B[eCut] - B[sCut], INT_MAX};
						}
						if (DP[sCut][eCut].second != INT_MAX) return DP[sCut][eCut];

						pair<int, int> ans = {INT_MAX, INT_MAX}, cut;
						for (int i = sCut + 1; i < eCut; i++) {
							cut = {cutRod(B, sCut, i).first + cutRod(B, i, eCut).first + B[eCut] - B[sCut], i};
							ans = min(ans, cut);
						}
						return DP[sCut][eCut] = ans;
					}
					vector<int> Solution::rodCut(int A, vector<int> &B) {
						B.push_back(0);
						B.push_back(A);
						int n = B.size();
						DP = vector<vector<pair<int, int>>>(n, vector<pair<int, int>>(n, {INT_MAX, INT_MAX}));
						sort(B.begin(), B.end());

						cutRod(B, 0, B.size() - 1);

						vector<int> order;
						fill(order, 0, B.size() - 1, B);
						return order;
					}
				}
				{
					//Itterative
					void fill(vector<int> &ans, vector<vector<long long>> &cut, int S, int E, vector<int> &A) {
						if (S + 1 >= E)return;
						int x = cut[S][E];
						ans.push_back(A[x]);
						fill(ans, cut, S, x, A);
						fill(ans, cut, x, E, A);
					}
					vector<int> Solution::rodCut(int A, vector<int> &B) {
						B.push_back(0);
						B.push_back(A);
						sort(B.begin(), B.end());
						int n = B.size();
						vector<vector<long long>> dp(n, vector<long long> (n, 0)), cut = dp;
						for (int i = n - 1; i >= 0; i--) {
							for (int j = i + 2; j < n; j++) {
								int x = i + 1;
								for (int k = i + 2; k < j; k++) {
									if (dp[i][k] + dp[k][j] < dp[i][x] + dp[x][j]) {
										x = k;
									}
								}
								dp[i][j] = B[j] - B[i] + dp[i][x] + dp[x][j];
								cut[i][j] = x;
							}
						}
						vector<int> S;
						fill(S, cut, 0, B.size() - 1, B);
						return S;
					}
				}
			}
			{
				/*
				Max Rectangle in Binary Matrix
				https://www.interviewbit.com/problems/max-rectangle-in-binary-matrix/

				#Good #VeryGood #Revise
				*/
				{
					//Using largest renctangle in Histogram
					int Solution::maximalRectangle(vector<vector<int> > &A) {
						int n = A.size(), m = A[0].size();
						vector<vector<int>> L(n, vector<int> (m, 0));
						for (int i = 0; i < n; i++) {
							for (int j = m - 1; j >= 0; j--) {
								if (A[i][j] != 0) {
									if (j == m - 1) {
										L[i][j] = 1;
									} else {
										L[i][j] = L[i][j + 1] + 1;
									}
								}
							}
						}
						int ans = 0;

						for (int j = 0; j < m; j++) {
							stack<int> S, T;
							vector<int> left(n), right(n, n - 1);

							for (int i = 0; i < n; i++) {
								while (!S.empty() && L[S.top()][j] >= L[i][j]) {
									S.pop();
								}
								if (!S.empty()) {
									left[i] = S.top() + 1;
								}
								S.push(i);
							}
							for (int i = n - 1; i >= 0; i--) {
								while (!T.empty() && L[T.top()][j] >= L[i][j]) {
									T.pop();
								}
								if (!T.empty()) {
									right[i] = T.top() - 1;
								}
								T.push(i);
							}
							for (int i = 0; i < n; i++) {
								ans = max(ans, L[i][j] * (right[i] - left[i] + 1));
							}
						}
						return ans;
					}
				}
				{
					//Itterative
					// Assume rectangle ending at i, j
					int Solution::maximalRectangle(vector<vector<int> > &A) {
						int n = A.size(), m = A[0].size(), ans = 0;
						vector<vector<int>> dp(n, vector<int>(m));
						for (int i = 0; i < n; i++) {
							for (int j = 0; j < m; j++) {
								if (A[i][j]) {
									dp[i][j] = (j == 0 ? 0 : dp[i][j - 1]) + 1;
									int width = dp[i][j];
									for (int k = i; k >= 0; k--) {
										width = min(width, dp[k][j]);
										ans = max(ans, width * (i - k + 1));
									}
								}
							}
						}
						return ans;
					}
				}
			}
			{
				/*
				Queen Attack
				https://www.interviewbit.com/problems/queen-attack/

				#Good
				*/
				void dfs(vector<string> &A, vector<vector<int>> &dp, int i, int j, int xs, int ys) {
					if (i<0 or i >= A.size() or j<0 or j >= A[0].size()) return;
					dp[i][j]++;
					if (A[i][j] == '0') dfs(A, dp, i + xs, j + ys, xs, ys);
				}

				vector<vector<int> > Solution::queenAttack(vector<string> &A) {
					int n = A.size(), m = A[0].size();
					vector<vector<int>> dp(n, vector<int>(m, 0));
					const int dx[] = { -1, 0, 1, 0, -1, -1, 1, 1};
					const int dy[] = {0, 1, 0, -1, -1, 1, -1, 1};
					for (int i = 0; i < n; ++i) {
						for (int j = 0; j < m; ++j) {
							if (A[i][j] == '1') {
								for (int k = 0; k < 8; ++k) {
									dfs(A, dp, i + dx[k], j + dy[k], dx[k], dy[k]);
								}
							}
						}
					}
					return dp;
				}
			}
		}
		// Suffix / prefix DP
		{
			{
				/*
				Sub Matrices with sum Zero
				https://www.interviewbit.com/problems/sub-matrices-with-sum-zero/

				#Good #Revise #VeryGood
				*/
				int countzero(vector<int> &A) {
					unordered_map<int, int> M;
					M[0] = 1;
					int sum = 0, count = 0;
					for (int i = 0; i < A.size(); i++) {
						sum += A[i];
						if (M.find(sum) != M.end())count += M[sum];
						M[sum]++;
					}
					return count;
				}
				int Solution::solve(vector<vector<int> > &A) {
					int count = 0;
					for (int i = 0; i < A.size(); i++) {
						vector<int> temp(A[0].size(), 0); // sum of coloumn
						for (int j = i; j < A.size(); j++) {
							for (int k = 0; k < A[0].size(); k++) {
								temp[k] += A[j][k];
							}
							// sum of coloumn starting from i to j
							count += countzero(temp);
						}
					}
					return count;
				}
			}
			{
				/*
				Coin Sum Infinite
				https://www.interviewbit.com/problems/coin-sum-infinite/

				#Good because of memory limit
				*/
				int Solution::coinchange2(vector<int> &A, int B) {
					int n = A.size();
					vector<int> dp(B + 1);
					dp[0] = 1;
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= B; j++) {
							if (j >= A[i - 1]) dp[j] += dp[j - A[i - 1]];
							dp[j] %= 1000007;
						}
					}
					return dp[B];
				}
			}
			{
				/*
				Max Product Subarray
				https://www.interviewbit.com/problems/max-product-subarray/

				#Good
				*/
				int Solution::maxProduct(const vector<int> &A) {
					int mx = INT_MIN;
					int res = 1;
					int n = A.size();
					for (int i = 0; i < n; i++)
					{
						res *= A[i];
						mx = max(mx, res);
						if (res == 0)res = 1;
					}
					res = 1;
					for (int i = n - 1; i >= 0; i--)
					{
						res *= A[i];
						mx = max(mx, res);
						if (res == 0)res = 1;
					}
					return mx;
				}
			}
			{
				/*
				Arrange II
				https://www.interviewbit.com/problems/arrange-ii/

				#Good
				*/
				void count(string & S, vector<int> &W, vector<int> &B) {
					int w = 0, b = 0;
					for (char s : S) {
						if (s == 'W')w++;
						else b++;
						W.push_back(w);
						B.push_back(b);
					}
				}
				int Solution::arrange(string A, int B) {
					int n = A.length();
					if (n < B)return -1;
					if (n == B)return 0;

					vector<int> W = {0}, Bl = W;
					count(A, W, Bl);
					vector<vector<int>> dp(B + 1, vector<int> (n + 1, 0));
					for (int j = 1; j <= n; j++) {
						dp[1][j] = W[j] * Bl[j];
					}
					for (int i = 2; i <= B; i++) {
						for (int j = i + 1; j <= n; j++) {
							dp[i][j] = INT_MAX;
							for (int k = j - 1; k >= i - 1; k--) {
								dp[i][j] = min(dp[i][j], dp[i - 1][k] + (W[j] - W[k]) * (Bl[j] - Bl[k]));
							}
						}
					}
					return dp[B][n];
				}

			}

		}
		// Derived DP
		{
			{
				/*
				Chain of Pairs
				https://www.interviewbit.com/problems/chain-of-pairs/
				*/
				int Solution::solve(vector<vector<int> > &A) {
					int n = A.size();
					vector<int> dp(n, 1);
					for (int i = 0; i < n; i++) {
						for (int j = i - 1; j >= 0; j--) {
							if (A[i][0] > A[j][1]) {
								dp[i] = max(dp[i], dp[j] + 1);
							}
						}
					}
					return *max_element(dp.begin(), dp.end());
				}
				{
					/*
					Max Sum Without Adjacent Elements
					https://www.interviewbit.com/problems/max-sum-without-adjacent-elements/
					*/
					int Solution::adjacent(vector<vector<int> > &Z) {
						int n = Z[0].size();
						vector<int> A(n + 1, 0);
						for (int i = 1; i <= n; i++) {
							A[i] = max(Z[0][i - 1], Z[1][i - 1]);
						}
						for (int i = 2; i <= n; i++) {
							A[i] = max(A[i - 1], A[i - 2] + A[i]);
						}
						return A[n];
					}
				}
				{
					/*
					Merge elements
					https://www.interviewbit.com/problems/merge-elements/
					*/
					int cost(vector<int> &A, int s, int e, vector<int> &sum, vector<vector<int>> &dp) {
						if (dp[s][e] != -1)return dp[s][e];
						int ans = INT_MAX;
						for (int i = s; i < e; i++) {
							ans = min(ans, cost(A, s, i, sum, dp) + cost(A, i + 1, e, sum, dp));
						}
						ans += sum[e + 1] - sum[s];
						dp[s][e] = ans;
						return ans;
					}
					int Solution::solve(vector<int> &A) {
						int n = A.size();
						vector<int> sum(n + 1, 0);
						vector<vector<int>> dp(n, vector<int> (n, -1));
						for (int i = 1; i <= n; i++) {
							dp[i - 1][i - 1] = 0;
							sum[i] = sum[i - 1] + A[i - 1];
						}
						return cost(A, 0, n - 1, sum, dp);
					}
				}
			}

		}
		// Knapsack
		{
			{
				/*
				Flip Array
				https://www.interviewbit.com/problems/flip-array/

				#Good #Tricky
				*/
				int Solution::solve(const vector<int> &A) {
					int n = A.size(), sum = 0;
					for (int x : A) sum += x;
					sum /= 2;
					vector<vector<pair<int, int>>> dp(n + 1, vector<pair<int, int>>(sum + 1)); //pair is {weight, -steps}
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= sum; j++) {
							dp[i][j] = dp[i - 1][j];
							if (j >= A[i - 1]) {
								auto x = dp[i - 1][j - A[i - 1]];
								dp[i][j] = max(dp[i][j], {x.first + A[i - 1], x.second - 1});
							}
						}
					}
					return -dp[n][sum].second;
				}
			}
			{
				/*
				Tushar's Birthday Party
				https://www.interviewbit.com/problems/tushars-birthday-party/
				*/
				int Solution::solve(const vector<int> &A, const vector<int> &B, const vector<int> &C) {
					int n = *max_element(A.begin(), A.end()); //max capa of a person
					int m = B.size(); //number of dishes
					vector<vector<long long>> M(n + 1, vector<long long>(m + 1, INT_MAX));
					// capacity of a person on row (ranging from 1 to maxcapa)
					// dish size on coloumn
					for (int j = 0; j < m + 1; j++) {
						M[0][j] = 0;
					}
					for (int i = 1; i < n + 1; i++) {
						for (int j = 1; j < m + 1; j++) {
							if (B[j - 1] <= i)
								M[i][j] = min( C[j - 1] + M[i - B[j - 1]][j], M[i][j - 1]);
							else
								M[i][j] = M[i][j - 1];
						}
					}
					int ans = 0;
					for (int i = 0; i < A.size(); i++) {
						ans += M[A[i]][m];
					}
					return ans;
				}
			}
			{
				/*
				0-1 Knapsack
				https://www.interviewbit.com/problems/0-1-knapsack/

				#Famous
				*/
				int Solution::solve(vector<int> &A, vector<int> &B, int C) {
					int n = A.size();
					vector<vector<int>> dp(n + 1, vector<int> (C + 1, 0));
					for (int i = 1; i <= n; i++) {
						for (int j = 1; j <= C; j++) {
							dp[i][j] = dp[i - 1][j];
							if (B[i - 1] <= j) {
								dp[i][j] = max(dp[i][j], dp[i - 1][j - B[i - 1]] + A[i - 1]);
							}
						}
					}
					return dp[n][C];
				}
			}
			{
				/*
				Equal Average Partition
				https://www.interviewbit.com/problems/equal-average-partition/
				*/
			}
		}
		// Breaking words
		{
			{
				/*
				Word Break
				https://www.interviewbit.com/problems/word-break/

				#Good #Famous
				*/
				bool check(string S, unordered_map<string, bool> &M, set<string> &BS) {
					if (M.find(S) != M.end())
						return M[S];
					for (int i = 1; i <= S.length(); i++) {
						if (BS.find(S.substr(0, i)) != BS.end()) {
							if (check(S.substr(i), M, BS)) {
								M[S] = true;
								return true;
							}
						}
					}
					M[S] = false;
					return false;
				}
				int Solution::wordBreak(string A, vector<string> &B) {
					set<string> BS;
					for (auto s : B)BS.insert(s);
					unordered_map<string, bool> M;
					M[""] = 1;
					if (check(A, M, BS))return 1;
					return 0;
				}
			}
			{
				/*
				Word Break II
				https://www.interviewbit.com/problems/word-break-ii/
				*/
				unordered_map<string, vector<string>> M;
				set<string> BS;

				vector<string> helper(string A) {
					if (M.find(A) != M.end()) {
						return M[A];
					}
					vector<string> S;
					for (int i = 1; i <= A.length(); i++) {
						string split = A.substr(0, i);
						if (BS.find(split) != BS.end()) {
							vector<string> T = helper(A.substr(i));
							for (auto s : T) {
								if (s.length())
									S.push_back(split + " " + s);
								else
									S.push_back(split);
							}
						}
					}
					M[A] = S;
					return S;
				}

				vector<string> Solution::wordBreak(string A, vector<string> &B) {
					M.clear();
					BS.clear();
					for (auto s : B)BS.insert(s);
					M[""] = vector<string>(1, "");
					return helper(A);
				}

			}
			{
				/*
				Palindrome Partitioning II
				https://www.interviewbit.com/problems/palindrome-partitioning-ii/
				*/
				vector<vector<int>> dp;
				vector<vector<int>> M;
				bool palidrome(string & A, int s, int len) {
					if (M[s][len] != -1)return M[s][len];
					string Z = A.substr(s, len);
					for (int i = 0; i < len / 2; i++) {
						if (Z[i] != Z[len - i - 1]) {
							M[s][len] = 0;
							return 0;
						}
					}
					M[s][len] = 1;
					return 1;
				}
				int helper(string & A, int s, int len) {
					if (dp[s][len] != -1)return dp[s][len];
					if (palidrome(A, s, len)) {
						dp[s][len] = 0;
						return 0;
					}
					int count = INT_MAX;
					for (int i = 1; i < len; i++) {
						count = min(count, helper(A, s, i) + 1 + helper(A, s + i, len - i));
					}
					dp[s][len] = count;
					return count;
				}
				int Solution::minCut(string A) {
					int n = A.length();
					dp = vector<vector<int>> (n, vector<int>(n + 1, -1));
					M = vector<vector<int>> (n, vector<int>(n + 1, -1));
					return helper(A, 0, n);
				}
			}
		}
		// Multiply DP
		{
			{
				/*
				Unique Binary Search Trees II
				https://www.interviewbit.com/problems/unique-binary-search-trees-ii/
				*/
				unordered_map<int, int> M;

				int tree(int A) {
					if (M.find(A) != M.end()) {
						return M[A];
					}
					int count = 0;
					for (int i = 0; i <= A - 1; i++) {
						if (i == 0 || i == A - 1)
							count += tree(A - 1);
						else
							count += (tree(i) * tree(A - i - 1));
					}
					M[A] = count;
					return count;
				}

				int Solution::numTrees(int A) {
					M.clear();
					M[0] = 0;
					M[1] = 1;
					return tree(A);
				}
			}
			{
				/*
				Count Permutations of BST
				https://www.interviewbit.com/problems/count-permutations-of-bst/
				*/
			}
		}
	}
// Tree
	{
		// BST Travel
		{
			{
				/*
				Valid BST from Preorder
				https://www.interviewbit.com/problems/valid-bst-from-preorder/

				#Famous
				*/
				int Solution::solve(vector<int> &A) {
					stack<int> s;
					int root = INT_MIN;
					for (int i = 0; i < A.size(); i++) {
						if (A[i] < root) return false;
						while (!s.empty() && A[i] > s.top()) {
							root = s.top();
							s.pop();
						}
						s.push(A[i]);
					}
					return true;
				}
			}
			{
				/*
				Kth Smallest Element In Tree
				https://interviewbit.com/problems/kth-smallest-element-in-tree/

				#Famous
				*/
				int num, ans;
				void trav(TreeNode * A, int &B) {
					if (!A)return;
					trav(A->left, B);
					num++;
					if (num == B) {
						ans = A->val;
						return;
					}
					trav(A->right, B);
				}
				int Solution::kthsmallest(TreeNode * A, int B) {
					num = 0;
					trav(A, B);
					return ans;
				}
			}
			{
				/*
				2-Sum Binary Tree
				https://www.interviewbit.com/problems/2sum-binary-tree/

				#Tricky
				*/
				stack<TreeNode*> Great, Small;

				TreeNode * nextg() {
					stack<TreeNode*> &Q = Great;
					TreeNode* ret = Q.top(), *curr = Q.top()->left;
					Q.pop();
					while (curr) {
						Q.push(curr);
						curr = curr->right;
					}

					return ret;
				}
				TreeNode * nexts() {
					stack<TreeNode*> &Q = Small;
					TreeNode* ret = Q.top(), *curr = Q.top()->right;
					Q.pop();
					while (curr) {
						Q.push(curr);
						curr = curr->left;
					}

					return ret;
				}
				int Solution::t2Sum(TreeNode * A, int B) {
					stack<TreeNode*> Q1, Q2;
					swap(Great, Q1);
					swap(Small, Q2);

					TreeNode* C = A;
					while (A) {
						Great.push(A);
						A = A->right;
					}
					while (C) {
						Small.push(C);
						C = C->left;
					}

					TreeNode* S = nexts(), *G = nextg();
					while (S && G && (S->val <= G->val) && S != G) {
						int temp = S->val + G->val;
						if (temp == B) {
							return 1;
						} else if (temp > B) {
							G = nextg();
						} else {
							S = nexts();
						}
					}
					return 0;
				}
			}
			{
				/*
				BST Iterator
				https://www.interviewbit.com/problems/bst-iterator/

				#Good
				*/
				stack<TreeNode*> Q;
				BSTIterator::BSTIterator(TreeNode * root) {
					stack<TreeNode*> empty;
					swap( Q, empty );
					while (root) {
						Q.push(root);
						root = root->left;
					}
				}

				/** @return whether we have a next smallest number */
				bool BSTIterator::hasNext() {
					if (Q.empty())return 0;
					return 1;
				}

				/** @return the next smallest number */
				int BSTIterator::next() {
					TreeNode *curr = Q.top();
					int ret = curr->val;
					Q.pop();
					curr = curr->right;
					while (curr) {
						Q.push(curr);
						curr = curr->left;
					}
					return ret;
				}
			}
		}
		// Trie
		{
			{
				/*
				Xor Between Two Arrays!
				https://www.interviewbit.com/problems/xor-between-two-arrays/

				#Good
				*/
				struct trie{
					map<int, trie*> mp;
					bool isLast;
				};
				trie * getNode() {
					trie *temp = new trie;
					temp->isLast = false;
					return temp;
				}
				void insert(trie * root, int a) {
					trie *p = root;
					for (int i = 31; i >= 0; i--) {
						int l = (a >> i) & 1;
						if (!p->mp.count(l)) {
							p->mp[l] = getNode();
						}
						p = p->mp[l];
					}
					p->isLast = true;
				}
				int mx = INT_MIN;
				void search(trie * root, int num) {
					trie* p = root;
					int curr = 0;
					for (int i = 31; i >= 0; i--) {
						int l = ( (num >> i) & 1 ? 0 : 1);
						if (p->mp.count(l)) {
							curr <<= 1;
							curr |= 1;
							p = p->mp[l];
						}
						else {
							curr <<= 1;
							curr |= 0;
							p = p->mp[l ^ 1];
						}
					}
					mx = max(mx, curr);
				}
				int Solution::solve(vector<int> &A, vector<int> &B) {
					trie *root = getNode();
					for (auto i : A) {
						insert(root, i);
					}
					mx = INT_MIN;
					for (auto i : B) {
						search(root, i);
					}
					return mx;
				}
			}
			{
				/*
				Hotel Reviews
				https://www.interviewbit.com/problems/hotel-reviews/
				*/
				struct Node{
					char val;
					vector<Node*> child;
					Node(char c): val(c) {}
				};

				vector<string> extract(string & A) {
					vector<string> S;
					int len = 0, j = 0;
					for (int i = 0; i <= A.length(); i++) {
						if (i == A.length() || A[i] == '_') {
							S.push_back(A.substr(j, len));
							j = i + 1;
							len = 0;
						} else {
							len++;
						}
					}
					return S;
				}

				void make_trie(string & A, Node * head) {
					vector<string> words = extract(A);
					for (auto s : words) {
						Node *check = head;
						for (int i = 0; i <= s.length(); i++) {
							bool found = false;
							for (int j = 0; j < ((check->child).size()); j++) {
								if ((check->child)[j]->val == s[i]) {
									found = true;
									check = (check->child)[j];
									break;
								}
							}
							if (!found) {
								(check->child).push_back(new Node(s[i]));
								check = (check->child).back();
							}
						}
					}
				}
				bool find_inT(string A, Node * check) {
					for (int i = 0; i <= A.length(); i++) {
						bool found = false;
						for (int j = 0; j < ((check->child).size()); j++) {
							if ((check->child)[j]->val == A[i]) {
								found = true;
								check = (check->child)[j];
								break;
							}
						}
						if (!found) {
							return false;
						}
					}
					return true;
				}
				void count(vector<string> &B, vector<pair<int, int>> &P, Node * head) {
					for (int i = 0; i < B.size(); i++) {
						vector<string> words = extract(B[i]);
						int C = 0;
						for (int j = 0; j < words.size(); j++) {
							if (find_inT(words[j], head))C++;
						}
						P.push_back({C, i});
					}
				}
				bool comp(pair<int, int> A, pair<int, int> B) {
					return A.first > B.first;
				}
				vector<int> Solution::solve(string A, vector<string> &B) {
					Node *head = new Node('0');
					vector<int> S;
					make_trie(A, head);
					vector<pair<int, int>> P;
					count(B, P, head);
					stable_sort(P.begin(), P.end(), comp);
					for (auto p : P) {
						S.push_back(p.second);
					}
					return S;
				}
			}
			{
				/*
				Shortest Unique Prefix
				https://www.interviewbit.com/problems/shortest-unique-prefix/

				#Good
				*/
				struct tree{
					char val;
					bool fork = false;
					vector<tree*> child;
					tree(char c): val(c) {}
				};
				tree * findval(tree * root, char c) {
					for (int i = 0; i < root->child.size(); i++) {
						if (root->child[i]->val == c) {
							return root->child[i];
						}
					}
					tree* ret = new tree(c);
					root->child.push_back(ret);
					return ret;
				}
				void trie(string A, tree * root) {
					for (int i = 0; i <= A.length(); i++) {
						root = findval(root, A[i]);
					}
				}
				void check(string s, vector<string> &S, tree * root) {
					string A = "";
					for (int i = 0; i <= s.length(); i++) {
						root = findval(root, s[i]);
						A += root->val;
						if (root->fork == false) {
							S.push_back(A);
							return;
						}
					}
				}
				void checkfork(tree * root) {
					if (root->child.size() > 1) {
						root->fork = true;
					}
					for (auto child : root->child) {
						checkfork(child);
						root->fork = child->fork;
					}
				}
				vector<string> Solution::prefix(vector<string> &A) {
					vector<string> S;
					tree *root = new tree('0');
					for (auto s : A) {
						trie(s, root);
					}
					checkfork(root);
					for (auto s : A) {
						check(s, S, root);
					}
					return S;
				}
			}
		}
		//Simple Tree Ops
		{
			{
				/*
				Path to Given Node
				https://www.interviewbit.com/problems/path-to-given-node/hints/

				#Famous
				*/
				bool findnode(TreeNode* &A, const int &B, vector<int> &S) {
					if (!A) return false;
					S.push_back(A->val);
					if (A->val == B) return true;
					if (findnode(A->left, B, S) || findnode(A->right, B, S)) return true;

					S.pop_back();
					return false;
				}
				vector<int> Solution::solve(TreeNode * A, int B) {
					vector<int> S;
					findnode(A, B, S);
					return S;
				}
			}
			{
				/*
				Remove Half Nodes
				https://www.interviewbit.com/problems/remove-half-nodes/
				*/
				TreeNode* Solution::solve(TreeNode * A) {
					if (!A)return NULL;
					A->left = solve(A->left);
					A->right = solve(A->right);
					if ( ((A->left) && (A->right)) || (!(A->left) && !(A->right)) ) {
						return A;
					} else if (A->left) {
						return A->left;
					} else {
						return A->right;
					}
				}
			}
			{
				/*
				Balanced Binary Tree
				https://www.interviewbit.com/problems/balanced-binary-tree/
				*/
				pair<int, bool> isBalancedWithDepth(TreeNode * root) {
					if (root == NULL) return make_pair(0, true);
					pair<int, bool> left = isBalancedWithDepth(root->left);
					pair<int, bool> right = isBalancedWithDepth(root->right);
					int depth = max(right.first, left.first) + 1;
					bool isbalanced = right.second && left.second && (abs(right.first - left.first) < 2);
					return make_pair(depth, isbalanced);
				}
				int Solution::isBalanced(TreeNode * root) {
					if (root == NULL) return true;
					return isBalancedWithDepth(root).second;
				}
			}
			{
				/*
				Maximum Edge Removal
				https://www.interviewbit.com/problems/maximum-edge-removal/
				*/
				int dfs(vector<int> A[], int start, int &ans, int parent) {
					int child = 0;
					for (auto node : A[start]) {
						if (node == parent) continue;
						int childNodes = dfs(A, node, ans, start);
						child += childNodes;
						if (childNodes % 2 == 0) ans += 1;
					}
					return child + 1;
				}
				int Solution::solve(int A, vector<vector<int> > &B) {
					int ans = 0;
					vector<int> arr[A];
					for (auto i : B) {
						arr[i[0] - 1].push_back(i[1] - 1);
						arr[i[1] - 1].push_back(i[0] - 1);
					}
					dfs(arr, 0, ans, 0);
					return ans;
				}
			}
		}
		// 2 Trees
		{
			{
				/*
				Merge two Binary Tree
				https://www.interviewbit.com/problems/merge-two-binary-tree/
				*/
				TreeNode* Solution::solve(TreeNode * A, TreeNode * B) {
					int val = 0;

					if (!A)return B;
					else val += A->val;

					if (!B)return A;
					else val += B->val;

					TreeNode* T = new TreeNode(val);
					T->left = solve(A->left, B->left);
					T->right = solve(A->right, B->right);
					return T;
				}
			}
			{
				/*
				Symmetric Binary Tree
				https://www.interviewbit.com/problems/symmetric-binary-tree/

				#Famous #Good
				*/
				int check(TreeNode * A, TreeNode * B) {
					if (!A && B || !B && A)return 0;
					else if (!A && !B) return 1;
					else if (A->val != B->val) return 0;
					else if (check(A->left, B->right) == 0) return 0;
					else if (check(A->right, B->left) == 0) return 0;
					return 1;
				}
				int Solution::isSymmetric(TreeNode * A) {
					return check(A->left, A->right);
				}
			}
			{
				/*
				Identical Binary Trees
				https://www.interviewbit.com/problems/identical-binary-trees/
				*/
				int Solution::isSameTree(TreeNode * A, TreeNode * B) {
					if (!A && B || A && !B) return 0;
					if (!A && !B) return 1;
					if (A->val != B->val)return 0;
					return isSameTree(A->left, B->left) && isSameTree(A->right, B->right);
				}
			}
		}
		// Tree Construction
		{
			{
				/*
				Inorder Traversal of Cartesian Tree
				https://www.interviewbit.com/problems/inorder-traversal-of-cartesian-tree
				*/
				TreeNode * build(vector<int> &A, int i, int j) {
					if (i > j)return NULL;
					int max_i = i;
					for (int m = i; m <= j; m++) {
						if (A[max_i] < A[m]) {
							max_i = m;
						}
					}
					TreeNode *root = new TreeNode(A[max_i]);
					root->left = build(A, i, max_i - 1);
					root->right = build(A, max_i + 1, j);
					return root;
				}
				TreeNode* Solution::buildTree(vector<int> &A) {
					return build(A, 0, A.size() - 1);
				}
			}
			{
				/*
				Sorted Array To Balanced BST
				https://www.interviewbit.com/problems/sorted-array-to-balanced-bst/hints/
				*/
				TreeNode * sortedArrayToBSTQ(const vector<int>& arr, int start, int end)  {
					if (start > end) return NULL;

					int mid = (start + end) / 2;
					TreeNode *root = new TreeNode(arr[mid]);
					root->left = sortedArrayToBSTQ(arr, start, mid - 1);
					root->right = sortedArrayToBSTQ(arr, mid + 1, end);
					return root;
				}
				TreeNode* Solution::sortedArrayToBST(const vector<int> &A) {
					TreeNode *root = sortedArrayToBSTQ(A, 0, A.size() - 1);
					return root;
				}
			}
			{
				/*
				Construct Binary Tree From Inorder And Preorder
				https://www.interviewbit.com/problems/construct-binary-tree-from-inorder-and-preorder/

				#Famous #Revise
				*/
				TreeNode * build(vector<int> &Preorder, vector<int> &Inorder, int a, int b, int root) {
					if (a > b)return NULL;
					int i = a;
					while (Preorder[root] != Inorder[i]) i++;
					TreeNode* S = new TreeNode(Inorder[i]);
					S->left = build(Preorder, Inorder, a, i - 1, root + 1);
					S->right = build(Preorder, Inorder, i + 1, b, root + i - a + 1);
					return S;
				}
				TreeNode* Solution::buildTree(vector<int> &Preorder, vector<int> &Inorder) {
					return build(Preorder, Inorder, 0, Preorder.size() - 1, 0);
				}
			}
			{
				/*
				Binary Tree From Inorder And Postorder
				https://www.interviewbit.com/problems/binary-tree-from-inorder-and-postorder/

				#Famous #Revise
				*/
				TreeNode * buildTree(vector<int>& inorder, vector<int>& postorder) {
					return build(inorder, postorder, 0, inorder.size() - 1, 0, postorder.size() - 1);
				}

				TreeNode * build(vector<int>& inorder, vector<int>& postorder, int inStart, int inEnd, int postStart, int postEnd) {
					if (inStart > inEnd) return nullptr;
					TreeNode* root = new TreeNode(postorder[postEnd]);
					int index = inStart;
					while (inorder[index] != root->val) index++;
					int leftLen = index - inStart;
					root->left = build(inorder, postorder, inStart, index - 1, postStart, postStart + leftLen - 1);
					root->right = build(inorder, postorder, index + 1, inEnd, postStart + leftLen, postEnd - 1);
					return root;
				}
			}
		}
		//Traversal
		{
			{
				/*
				Vertical Order traversal of Binary Tree
				https://www.interviewbit.com/problems/vertical-order-traversal-of-binary-tree/
				*/
				vector<vector<int> > Solution::verticalOrderTraversal(TreeNode * root) {
					// Base case
					vector<vector<int> >ans;
					if (root == NULL)return ans;
					// Create a map and store vertical oder in
					// map using function getVerticalOrder()
					map < int, vector<int> > m;
					int hd = 0;

					// Create queue to do level order traversal.
					// Every item of queue contains node and
					// horizontal distance.
					queue<pair<TreeNode*, int> > que;
					que.push(make_pair(root, hd));

					while (!que.empty())
					{
						// pop from queue front
						pair<TreeNode *, int> temp = que.front();
						que.pop();
						hd = temp.second;
						TreeNode* node = temp.first;

						// insert this node's data in vector of hash
						m[hd].push_back(node->val);

						if (node->left != NULL)
							que.push(make_pair(node->left, hd - 1));
						if (node->right != NULL)
							que.push(make_pair(node->right, hd + 1));
					}

					// Traverse the map and print nodes at
					// every horigontal distance (hd)
					map< int, vector<int> > :: iterator it;
					for (it = m.begin(); it != m.end(); it++)
					{
						ans.push_back(it->second);
					}
					return ans;
				}
				{
					/*
					Diagonal Traversal
					https://www.interviewbit.com/problems/diagonal-traversal/
					*/
					void traverse(TreeNode * A, int level, map<int, vector<int>> &M) {
						if (!A) return;
						M[level].push_back(A->val);
						traverse(A->left, level + 1, M);
						traverse(A->right, level, M);
					}
					vector<int> Solution::solve(TreeNode * A) {
						vector<int> ans;
						map<int, vector<int>> M;
						traverse(A, 0, M);
						for (auto [n, x] : M) {
							for (auto a : x) {
								ans.push_back(a);
							}
						}
						return ans;
					}
				}
				{
					/*
					Inorder Traversal
					https://www.interviewbit.com/problems/inorder-traversal/
					*/
					void traverse(vector<int> &S, TreeNode * A) {
						if (!A)return;
						traverse(S, A->left);
						S.push_back(A->val);
						traverse(S, A->right);
					}
					vector<int> Solution::inorderTraversal(TreeNode * A) {
						vector<int> S;
						traverse(S, A);
						return S;
					}
				}
				{
					/*
					Preorder Traversal
					https://www.interviewbit.com/problems/preorder-traversal/
					*/
					void traversal(TreeNode * A, vector<int> &S) {
						if (!A)return;
						S.push_back(A->val);
						traversal(A->left, S);
						traversal(A->right, S);
					}
					vector<int> Solution::preorderTraversal(TreeNode * A) {
						vector<int> S;
						traversal(A, S);
						return S;
					}
				}
				{
					/*
					Postorder Traversal
					https://www.interviewbit.com/problems/postorder-traversal/
					*/
					void traversal(TreeNode * A, vector<int> &S) {
						if (!A)return;
						traversal(A->left, S);
						traversal(A->right, S);
						S.push_back(A->val);
					}
					vector<int> Solution::postorderTraversal(TreeNode * A) {
						vector<int> S;
						traversal(A, S);
						return S;
					}
				}
			}
			//Level Order
			{
				{
					/*
					Right view of Binary tree
					https://www.interviewbit.com/problems/right-view-of-binary-tree/
					*/
					vector<int> Solution::solve(TreeNode * A) {
						queue<pair<int, TreeNode*>> Q;
						Q.push({0, A});
						vector<int> S;
						while (!Q.empty()) {
							auto [h, N] = Q.front();
							Q.pop();
							if (Q.empty() || Q.front().first > h) {
								S.push_back(N->val);
							}
							if (N->left)
								Q.push({h + 1, N->left});
							if (N->right)
								Q.push({h + 1, N->right});
						}
						return S;
					}
				}
				{
					/*
					Reverse Level Order
					https://www.interviewbit.com/problems/reverse-level-order/
					*/
					vector<int> Solution::solve(TreeNode * A) {
						vector<int> S;
						queue<TreeNode*> Q;
						Q.push(A);
						while (!Q.empty()) {
							if (Q.front()->right) {
								Q.push(Q.front()->right);
							}
							if (Q.front()->left) {
								Q.push(Q.front()->left);
							}
							S.push_back(Q.front()->val);
							Q.pop();
						}
						reverse(S.begin(), S.end());
						return S;
					}
				}
				{
					/*
					ZigZag Level Order Traversal BT
					https://www.interviewbit.com/problems/zigzag-level-order-traversal-bt/
					*/
					vector<vector<int> > Solution::zigzagLevelOrder(TreeNode * A) {
						vector<vector<int>> S;
						queue<pair<int, TreeNode*>> Q;
						Q.push({0, A});
						int p = -1;
						while (Q.size()) {
							auto [h, N] = Q.front();
							Q.pop();
							if (N->left) {
								Q.push({h + 1, N->left});
							}
							if (N->right) {
								Q.push({h + 1, N->right});
							}
							if (h > p) {
								S.push_back(vector<int> {N->val});
								p = h;
							} else {
								S[h].push_back(N->val);
							}
						}
						for (int i = 0; i < S.size(); i++) {
							if (i % 2) {
								reverse(S[i].begin(), S[i].end());
							}
						}
						return S;
					}
				}
				{
					/*
					Populate Next Right Pointers Tree
					https://www.interviewbit.com/problems/populate-next-right-pointers-tree/

					#Good
					*/
					void Solution::connect(TreeLinkNode * root) {
						if (root == nullptr)
							return;

						queue<TreeLinkNode*> q;
						q.push(root);

						while (!q.empty()) {
							int size = q.size();
							TreeLinkNode *prev = nullptr;

							for (int i = 0; i < size; ++i) {
								TreeLinkNode *current = q.front();
								q.pop();

								if (prev != nullptr)
									prev->next = current;

								prev = current;

								if (current->left != nullptr)
									q.push(current->left);
								if (current->right != nullptr)
									q.push(current->right);
							}
							prev->next = nullptr; // Set the next pointer of the last node in the level to NULL
						}
					}
				}
				{
					/*
					Cousins in Binary Tree
					https://www.interviewbit.com/problems/cousins-in-binary-tree/
					*/
					vector<int> Solution::solve(TreeNode * A, int B) {
						vector<int> cousins;
						if (A == nullptr)
							return cousins;

						queue<TreeNode*> q;
						q.push(A);
						bool found = false;

						while (!q.empty() && !found) {
							int size = q.size();
							cousins.clear(); // Clearing the cousins for each level

							for (int i = 0; i < size; ++i) {
								TreeNode* curr = q.front();
								q.pop();

								if ((curr->left && curr->left->val == B) || (curr->right && curr->right->val == B)) {
									found = true;
								} else {
									if (curr->left) q.push(curr->left);
									if (curr->right) q.push(curr->right);
								}
							}

							if (found) {
								while (!q.empty()) {
									cousins.push_back(q.front()->val);
									q.pop();
								}
							}
						}

						return cousins;
					}
				}
			}
			//Root to Leaf
			{
				{
					/*
					Max Depth of Binary Tree
					https://www.interviewbit.com/problems/max-depth-of-binary-tree/
					*/
					int Solution::maxDepth(TreeNode * A) {
						if (A == NULL)return 0;
						else {
							return max(maxDepth(A->left), maxDepth(A->right)) + 1;
						}
					}
				}
				{
					/*
					Sum Root to Leaf Numbers
					https://www.interviewbit.com/problems/sum-root-to-leaf-numbers/
					*/
					bool isleaf(TreeNode * A) {
						if (!A->left && !A->right)return true;
						return false;
					}
					void traverse(TreeNode * A, int x, int &ans) {
						if (!A)return;
						x = (x * 10 + A->val) % 1003;
						if (isleaf(A)) {
							ans = (ans + x) % 1003;
							return;
						}
						traverse(A->left, x, ans);
						traverse(A->right, x, ans);
					}
					int Solution::sumNumbers(TreeNode * A) {
						int ans = 0;
						traverse(A, 0, ans);
						return ans;
					}
				}
				{
					/*
					Path Sum
					https://www.interviewbit.com/problems/path-sum/
					*/
					bool isLeaf(TreeNode * A) {
						if (!A->left && !A->right) {
							return 1;
						}
						return 0;
					}
					int traverse(TreeNode * A, int &B, int &sum) {
						sum += A->val;
						if (sum == B && isLeaf(A))return 1;
						else if (sum >= B)return 0;

						if (A->left && traverse(A->left, B, sum)) {
							return 1;
						}
						if (A->right && traverse(A->right, B, sum)) {
							return 1;
						}

						sum -= A->val;
						return 0;
					}
					int Solution::hasPathSum(TreeNode * A, int B) {
						if (!A)return 0;
						int sum = 0;
						return traverse(A, B, sum);
					}
				}
				{
					/*
					Min Depth of Binary Tree
					https://www.interviewbit.com/problems/min-depth-of-binary-tree/
					*/
					int ans = INT_MAX;
					void traverse(TreeNode * A, int h) {
						if (!A)return;
						if (!A->left && !A->right) {
							ans = min(ans, h);
						}
						traverse(A->left, h + 1);
						traverse(A->right, h + 1);
					}
					int Solution::minDepth(TreeNode * A) {
						ans = INT_MAX;
						traverse(A, 1);
						return ans;
					}
				}
				{
					/*
					Root to Leaf Paths With Sum
					https://www.interviewbit.com/problems/root-to-leaf-paths-with-sum/
					*/
					bool isLeaf(TreeNode * A) {
						if (!A->left && !A->right) {
							return true;
						}
						return false;
					}
					void traverse(TreeNode * A, int &B, vector<vector<int>> &V, vector<int> &T, int &sum) {
						if (sum == B && isLeaf(A)) {
							V.push_back(T);
							return;
						}
						if (A->left) {
							sum += A->left->val;
							T.push_back(A->left->val);
							traverse(A->left, B, V, T, sum);
							T.pop_back();
							sum -= A->left->val;
						}
						if (A->right) {
							sum += A->right->val;
							T.push_back(A->right->val);
							traverse(A->right, B, V, T, sum);
							T.pop_back();
							sum -= A->right->val;
						}
					}
					vector<vector<int> > Solution::pathSum(TreeNode * A, int B) {
						vector<vector<int>> V;
						vector<int> T;
						if (!A)return V;
						int sum = A->val;
						T.push_back(A->val);
						traverse(A, B, V, T, sum);
						return V;
					}
				}
				{
					/*
					Burn a Tree
					https://www.interviewbit.com/problems/burn-a-tree/hints/
					*/
					{
						/*
						    ***********ADDITIONAL INFO*************
						    lDepth - maximum height of left subtree
						    rDepth - maximum height of right subtree
						    contains - stores true if tree rooted at current node contains the first burnt node
						    time - time to reach fire from the initally burnt leaf node to this node
						*/

						struct Info {
							int lDepth;
							int rDepth;
							bool contains;
							int time;
							Info()
							{
								lDepth = rDepth = 0;
								contains = false;

								time = -1;
							}
						};

						/*
						    Function to calculate time required to burn tree completely
						    node - address of current node
						    info - extra information about current node
						    target - node that is fired
						    res - stores the result
						*/

						void calcTime(TreeNode * node, Info & info, int target, int& res) {
							// Base case: if root is null
							if (node == NULL) {
								return;
							}

							// If current node is leaf
							if (node->left == NULL && node->right == NULL) {

								// If current node is the first burnt node
								if (node->val == target) {
									info.contains = true;
									info.time = 0;
								}
								return;
							}

							// Information about left child of root
							Info leftInfo;
							calcTime(node->left, leftInfo, target, res);

							// Information about right child of root
							Info rightInfo;
							calcTime(node->right, rightInfo, target, res);

							// If left subtree contains the fired node then time required to reach fire to current node will be (1 + time required for left child)
							info.time = (node->left && leftInfo.contains) ? (leftInfo.time + 1) : -1;

							// If right subtree contains the fired node then time required to reach fire to current node will be (1 + time required for right child)
							if (info.time == -1)
								info.time = (node->right && rightInfo.contains) ? (rightInfo.time + 1) : -1;

							// Storing(true or false) if the tree rooted at current node contains the fired node
							info.contains = ((node->left && leftInfo.contains) || (node->right && rightInfo.contains));

							// Calculate the maximum depth of left subtree
							info.lDepth = !(node->left) ? 0 : (1 + max(leftInfo.lDepth, leftInfo.rDepth));

							// Calculate the maximum depth of right subtree
							info.rDepth = !(node->right) ? 0 : (1 + max(rightInfo.lDepth, rightInfo.rDepth));

							// Calculating answer
							if (info.contains) {
								// If left subtree exists and it contains the fired node
								if (node->left && leftInfo.contains) {
									// calculate result
									res = max(res, info.time + info.rDepth);
								}

								// If right subtree exists and it contains the fired node
								if (node->right && rightInfo.contains) {
									// calculate result
									res = max(res, info.time + info.lDepth);
								}
							}
						}

						int Solution::solve(TreeNode * A, int B) {
							int res = 0;
							Info info;
							calcTime(A, info, B, res);
							return res;
						}

					}
				}
			}
			// Adhoc
			{
				{
					/*
					Invert the Binary Tree
					https://www.interviewbit.com/problems/invert-the-binary-tree/
					*/
					TreeNode* Solution::invertTree(TreeNode * A) {
						if (!A)return A;
						swap(A->left, A->right);
						invertTree(A->left);
						invertTree(A->right);
						return A;
					}
				}
				{
					/*
					Least Common Ancestor
					https://www.interviewbit.com/problems/least-common-ancestor/
					https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/

					#Famous #Revise
					*/
					TreeNode * lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q) {
						if (root == NULL || root == p || root == q) return root;
						TreeNode* left = lowestCommonAncestor(root->left, p, q);
						TreeNode* right = lowestCommonAncestor(root->right, p, q);
						return left == NULL ? right : left
					}
				}
				{
					/*
					Flatten Binary Tree to Linked List
					https://www.interviewbit.com/problems/flatten-binary-tree-to-linked-list/

					#Famous #Revise
					*/
					// Without While Loop
					{
						TreeNode * flat(TreeNode * A, TreeNode * parent) {
							if (!A) return parent;

							TreeNode *right = A->right;
							A->right = A->left;
							A->left = NULL;

							TreeNode *end = flat(A->right, A);
							end -> right = right;
							return flat(right, end);
						}
						TreeNode* Solution::flatten(TreeNode * A) {
							flat(A, A);
							return A;
						}
					}
					//With while loop
					{
						void flatten(TreeNode * root) {
							if (!root) return;
							TreeNode* right = root->right;
							root->right = root->left;
							root->left = nullptr;
							flatten(root->right);
							while (root->right) root = root->right;
							root->right = right;
							flatten(right);
						}
					}
				}
			}
		}
	}
// Heaps And Maps
	{
		//Heap
		{
			{
				/*
				Ways to form Max Heap
				https://www.interviewbit.com/problems/ways-to-form-max-heap/
				*/
				long long int combination(int n, int r, int mod) {
					int C[r + 1];
					memset(C, 0, sizeof(C));

					C[0] = 1;
					for (int i = 1; i <= n; i++)
					{
						for (int j = min(i, r); j > 0; j--)
							C[j] = (C[j] + C[j - 1]) % mod;
					}
					return C[r];
				}
				int Solution::solve(int A) {
					int n = A;
					if (A <= 1) {
						return 1;
					}
					else {
						int h = log2(n);
						int m = pow(2, h);
						int p = n + 1 - m;
						int l;
						if (p >= m / 2)
							l = m - 1;
						else
							l = m / 2 - 1 + p;
						int r = n - 1 - l;
						long long int x = combination(n - 1, l, 1000000007);
						long long int y = solve(l);
						long long int z = solve(r);
						return (((x * y) % 1000000007) * z) % 1000000007;
					}
				}
			}
			{
				/*
				N max pair combinations
				https://interviewbit.com/problems/n-max-pair-combinations/

				#Famous #Revise
				*/
				vector<int> Solution::solve(vector<int> &A, vector<int> &B) {
					sort(A.begin(), A.end(), greater<int>());
					sort(B.begin(), B.end(), greater<int>());
					int n = A.size();
					priority_queue<tuple<int, int, int> > pq;
					for (int i = 0; i < n; i++) {
						pq.push({A[i] + B[0], i , 0});
					}
					vector<int> ans;
					while (n--)
					{
						auto [sum, i , j] = pq.top();
						pq.pop();
						ans.push_back(sum);
						pq.push({A[i] + B[j + 1] , i , j + 1});
					}
					return ans;
				}
			}
			{
				/*
				K Largest Elements
				https://www.interviewbit.com/problems/k-largest-elements/
				*/
				vector<int> Solution::solve(vector<int> &A, int B) {
					priority_queue<int> M;
					for (int a : A) {
						M.push(a);
					}
					vector<int> S;
					for (int i = 0; i < B; i++) {
						S.push_back(M.top());
						M.pop();
					}
					return S;
				}
			}
			{
				/*
				Profit Maximisation
				https://www.interviewbit.com/problems/profit-maximisation/
				*/
				int Solution::solve(vector<int> &A, int B) {
					priority_queue<int> Q;
					for (int x : A) {
						Q.push(x);
					}
					int ans = 0;
					for (int i = 0; i < B; i++) {
						ans += Q.top();
						Q.push(Q.top() - 1);
						Q.pop();
					}
					return ans;
				}
			}
			{
				/*
				Merge K sorted arrays!
				https://www.interviewbit.com/problems/merge-k-sorted-arrays/hints/

				#Good #Revise
				*/
				typedef pair<int, pair<int, int>> ppi;

				vector<int> Solution::solve(vector<vector<int> > &A) {
					priority_queue<ppi, vector<ppi>, greater<ppi>> q;
					vector<int> res;
					for (int i = 0; i < A.size(); i++) q.push({A[i][0], {i, 0}});

					pair<int, int> pr;

					while (!q.empty())
					{
						res.push_back(q.top().first);
						pr = q.top().second;
						q.pop();

						if (pr.second + 1 < A[pr.first].size())
							q.push({A[pr.first][pr.second + 1], {pr.first, pr.second + 1}});
					}
					return res;
				}
			}
			{
				/*
				Magician and Chocolates
				https://www.interviewbit.com/problems/magician-and-chocolates/
				*/
				int Solution::nchoc(int A, vector<int> &B) {
					int mod = 1000000007;
					priority_queue<int> Q;
					for (int x : B) Q.push(x);
					long long ans = 0;
					while (A--) {
						int top = Q.top();
						Q.pop();
						Q.push(top / 2);
						ans += top;
						ans %= mod;
					}
					return ans;
				}
			}
			{
				/*
				Merge K Sorted Lists
				https://www.interviewbit.com/problems/merge-k-sorted-lists/

				#Solution
				*/
				{
					struct myStruct{
						ListNode* node;

						myStruct(ListNode * _node): node(_node) {}

						bool operator<(const myStruct & second) const {
							if (!second.node) return true;
							if (!node) return false;
							return node->val > second.node->val;
						}
					};

					ListNode* Solution::mergeKLists(vector<ListNode*> &A) {
						priority_queue<myStruct> Q;

						for (auto node : A) Q.push(myStruct(node));
						ListNode *head = new ListNode(0), *ans = head;
						while (!Q.empty()) {
							myStruct top = Q.top();
							Q.pop();
							if (!top.node) continue;
							head = head->next = top.node;
							Q.push(myStruct(top.node->next));
						}
						return ans->next;
					}
				}
			}
		}
		// Map
		{
			{
				/*
				Distinct Numbers in Window
				https://www.interviewbit.com/problems/distinct-numbers-in-window/
				*/
				vector<int> Solution::dNums(vector<int> &A, int B) {
					unordered_map<int, int> M;
					vector<int> ans(A.size() - B + 1, 0);
					for (int i = 0; i < B; i++) {
						M[A[i]]++;
					}
					ans[0] = M.size();
					for (int i = B; i < A.size(); i++) {
						if (A[i] != A[i - B]) {
							if (M[A[i - B]] == 1) {
								M.erase(A[i - B]);
							} else {
								M[A[i - B]]--;
							}
							M[A[i]]++;
						}
						ans[i - B + 1] = M.size();
					}
					return ans;
				}
			}
		}
	}
// Hashing
	{
		//Hash Search
		{
			{
				/*
				Colorful Number
				https://www.interviewbit.com/problems/colorful-number/
				*/
				int Solution::colorful(int A) {
					if (A < 10) return 1;
					set<int> s;
					vector<int> v;
					while (A) {
						int lastdigit = A % 10;
						A /= 10;
						for (auto &i : v) i *= lastdigit;
						v.push_back(lastdigit);
						for (auto i : v) {
							if (s.count(i)) return 0;
							else s.insert(i);
						}
					}
					return 1;
				}
			}
			{
				/*
				Largest Continuous Sequence Zero Sum
				https://www.interviewbit.com/problems/largest-continuous-sequence-zero-sum/
				*/
				vector<int> Solution::lszero(vector<int> &A) {
					unordered_map<int, int> M;
					int sum = 0, len = 0;
					vector<int> ind = {0, -1};

					for (int i = 0; i < A.size(); i++) {
						sum += A[i];
						if (M.find(sum) == M.end()) {
							M[sum] = i;
						}
						if (sum == 0) {
							len = i + 1;
							ind[0] = 0;
							ind[1] = i;
						} else if (i - M[sum] > len) {
							len = i - M[sum];
							ind[0] = M[sum] + 1;
							ind[1] = i;
						}
					}
					return vector<int> (A.begin() + ind[0], A.begin() + ind[1] + 1);
				}
			}
			{
				/*
				Longest Subarray Length: You need to find the length of the longest subarray having count of 1’s only one more than count of 0’s.
				https://www.interviewbit.com/problems/longest-subarray-length/

				#Famous #MustRevise #Revise
				*/
				int Solution::solve(vector<int> &A) {
					unordered_map<int, int> M;
					int sum = 0, ans = 0;
					for (int i = 0; i < A.size(); i++) {
						if (A[i]) sum++;
						else sum--;

						if (M.find(sum) == M.end()) {
							M[sum] = i;
						}
						if (sum == 1) {
							ans = i + 1;
						} else if (M.find(sum - 1) != M.end()) {
							ans = max(ans, (i - M[sum - 1]));
						}
					}
					return ans;
				}
			}
			{
				/*
				First Repeating element
				https://www.interviewbit.com/problems/first-repeating-element/
				*/
				int Solution::solve(vector<int> &A) {
					unordered_map<int, int> M;
					int ans = INT_MAX;
					for (int i = 0; i < A.size(); i++) {
						if (M.find(A[i]) == M.end()) {
							M[A[i]] = i;
						} else {
							ans = min(ans, M[A[i]]);
						}
					}
					return ans == INT_MAX ? -1 : A[ans];
				}
			}
			{
				/*
				2 Sum
				https://www.interviewbit.com/problems/2-sum/

				#Famous
				*/
				vector<int> Solution::twoSum(const vector<int> &A, int B) {
					map<int, int>mymap;

					for (int i = 0; i < A.size(); i++)
					{
						if (mymap.find(B - A[i]) != mymap.end())return{mymap[B - A[i]], i + 1};
						if (mymap.find(A[i]) == mymap.end())mymap[A[i]] = i + 1;
					}
					return {};
				}
			}
			{
				/*
				4 Sum
				https://www.interviewbit.com/problems/4-sum/

				#Famous
				*/
				// Perfect Solution
				{
					vector<vector<int> > Solution::fourSum(vector<int> &num, int target) {
						sort(num.begin(), num.end());
						unordered_map<int, set<pair<int, int>>> hash;
						set<vector<int>> ans;
						int n = num.size();
						for (int i = 0; i < n; i ++) {
							for (int j = i + 1; j < n; j ++) {
								int a = num[i] + num[j];
								if (hash.count(target - a)) {
									for (auto &p : hash[target - a]) {
										vector<int> b = {p.first, p.second, num[i], num[j]}; // as set for value was made for index [0, i-1] there are no overlaps
										ans.insert(b);
									}
								}
							}
							for (int j = 0; j < i; j ++) { //adding sum to hash before index i
								int a = num[j], b = num[i];
								hash[a + b].insert(make_pair(a, b));
							}
						}
						return vector<vector<int>>(ans.begin(), ans.end());
					}
				}
				//Okay Solution
				{
					vector<vector<int> > Solution::fourSum(vector<int> &A, int B) {
						sort(A.begin(), A.end());
						unordered_map<int, vector<pair<int, int>>> M;
						set<vector<int>> S;
						vector<vector<int> > ans;
						for (int i = 0; i < A.size(); i++) {
							for (int j = i + 1; j < A.size(); j++) {
								M[A[i] + A[j]].push_back({i, j});
							}
						}
						for (int k = 0; k < A.size(); k++) {
							for (int l = k + 1; l < A.size(); l++) {
								for (auto [i, j] : M[B - A[k] - A[l]]) {
									if (j < k) {
										S.insert({A[i], A[j], A[k], A[l]});
									}
								}
							}
						}
						for (auto x : S) ans.push_back(x);
						return ans;
					}
				}
			}
			{
				/*
				Valid Sudoku
				https://www.interviewbit.com/problems/valid-sudoku/

				#Famous
				*/
				int Solution::isValidSudoku(const vector<string> &A) {
					int row[9][9] = {0}, col[9][9] = {0}, box[9][9] = {0};

					for (int i = 0; i < 9; i++)
						for (int j = 0; j < 9; j++)
						{
							if (A[i][j] == '.')continue;
							int ind = int(A[i][j]) - 49;
							int bx = i - i % 3 + j / 3;
							if (row[i][ind] || col[j][ind] || box[bx][ind]) { return 0; }
							box[bx][ind] = row[i][ind] = col[j][ind] = 1;
						}
					return 1;
				}
			}
			{
				/*
				Diffk II
				https://www.interviewbit.com/problems/diffk-ii/
				*/
				int Solution::diffPossible(const vector<int> &A, int B) {
					unordered_map<int, int> M;
					for (int i = 0; i < A.size(); i++) {
						if (M.find(A[i] - B) == M.end() && M.find(A[i] + B) == M.end()) {
							M[A[i]] = i;
						} else {
							return 1;
						}
					}
					return 0;
				}
			}
			//Key Formation
			{

				{
					/*
					Equal: Given an array A of N integers, find the index of values that satisfy P + Q = R + S, where P, Q, R & S are integers values in the array
					https://www.interviewbit.com/problems/equal/
					*/
					vector<int> Solution::equal(vector<int> &vec) {
						int N = vec.size();
						// With every sum, we store the lexicographically first occuring pair of integers.
						map<int, pair<int, int> > M;
						vector<int> Ans;

						for (int R = 0; R < N; ++R) {
							for (int S = R + 1; S < N; ++S) {

								int sum = vec[R] + vec[S];

								if (M.find(sum) == M.end()) {
									M[sum] = make_pair(R, S);
									continue;
								}

								auto[P, Q] = M[sum];
								if (P != R && P != S && Q != R && Q != S) {

									vector<int> temp = {P, Q, R, S};

									if (Ans.size() == 0)
										Ans = temp;
									else
										Ans = min(Ans, temp);
								}
							}
						}

						return Ans;
					}
				}
				{
					/*
					Copy List with Random Pointer
					https://www.interviewbit.com/problems/copy-list/
					https://leetcode.com/problems/copy-list-with-random-pointer/

					#Famous
					*/
					class Solution {
					public:
						Node * copyRandomList(Node * head) {
							unordered_map<Node*, Node*> map;
							Node* ans = makeNextCopy(map, head);
							makeRandomCopy(ans, head, map);
							return ans;
						}

					private:
						Node * makeNextCopy(unordered_map<Node*, Node*> &map, Node * head) {
							Node *copy = new Node(0), *copy_itt = copy;
							while (head != NULL) {
								copy_itt->next = new Node(head->val);

								map[head] = copy_itt->next;

								copy_itt = copy_itt -> next;
								head = head->next;
							}
							return copy->next;
						}

						void makeRandomCopy(Node * copy, Node * head, unordered_map<Node*, Node*> &map) {
							while (head != NULL) {
								if (head->random != NULL) {
									copy->random = map[head->random];
								}
								copy = copy -> next;
								head = head -> next;
							}
						}
					};
				}
			}
			// Maths and Hashing
			{
				/*
				Check Palindrome!
				https://www.interviewbit.com/problems/check-palindrome/
				*/
				int Solution::solve(string A) {
					vector<int> M(26, 0);
					for (int i = 0; i < A.length(); i++) {
						M[A[i] - 'a']++;
					}
					int count = 0;
					for (int i = 0; i < 26; i++) {
						count += (M[i] % 2);
					}
					if (count > 1)return 0;
					return 1;
				}
			}
			{
				/*
				Fraction
				https://www.interviewbit.com/problems/fraction/
				*/
				string Solution::fractionToDecimal(int numerator, int denominator) {
					if (numerator == 0) return "0";
					if (denominator == 0) return "";
					string result = "";
					if ((numerator < 0) ^ (denominator < 0)) {
						result += "-";
					}
					long num = numerator, den = denominator;
					num = abs(num), den = abs(den);

					long res = num / den;
					result += to_string(res);

					long rem = (num % den) * 10;
					if (rem == 0) return result;

					map<long, int> mp;
					result += ".";
					while (rem != 0) {
						if (mp.find(rem) != mp.end()) {
							int beg = mp[rem];
							string part1 = result.substr(0, beg);
							string part2 = result.substr(beg, result.length() - beg);
							result = part1 + "(" + part2 + ")";
							return result;
						}
						mp[rem] = result.length();
						res = rem / den;
						result += to_string(res);
						rem = (rem % den) * 10;
					}
					return result;
				}
			}
			{
				/*
				Points on the Straight Line
				https://www.interviewbit.com/problems/points-on-the-straight-line/

				#Famous #Good
				*/
				int Solution::maxPoints(vector<int> &A, vector<int> &B) {
					if (A.size() <= 2) {
						return A.size();
					}
					int ans = 0;
					unordered_map<double, int> M;
					double slope;
					for (int i = 0; i < A.size(); i++) {
						int overlap = 0;
						for (int j = i + 1; j < A.size(); j++) {
							if (A[i] == A[j]) {
								if (B[i] == B[j]) {
									overlap++;
									continue;
								}
								slope = INT_MAX;
							} else {
								slope = (double)(B[i] - B[j]) / (double)(A[i] - A[j]);
							}
							M[slope]++;
						}
						ans = max(ans, overlap);
						for (auto m : M) {
							ans = max(ans, m.second + overlap);
						}
						M.clear();
					}
					return ans + 1;
				}
			}
			{
				/*
				Find the Duplicate Number
				https://leetcode.com/problems/find-the-duplicate-number/description/

				#Famous #Good
				*/
				int findDuplicate(vector<int>& nums) {
					int duplicate = -1, n = nums.size();
					for (int i = 0; i < n; i++) {
						int curr = abs(nums[i]);
						if (nums[curr - 1] < 0) {
							duplicate = curr;
							break;
						}
						nums[curr - 1] *= -1;
					}
					for (auto &num : nums) num = abs(num);
					return duplicate;
				}

			}
		}
		//Incremental Hash
		{
			{
				/*
				An Increment Problem
				https://www.interviewbit.com/problems/an-increment-problem/
				*/
				vector<int> Solution::solve(vector<int> &A) {
					unordered_map<int, set<int>> M;
					for (int i = 0; i < A.size(); i++) {
						int x = A[i];
						if (M.find(x) != M.end()) {
							int j = *M[x].begin();
							A[j]++;
							M[A[j]].insert(j);
							M[x].erase(M[x].begin());
						}
						M[x].insert(i);
					}
					return A;
				}
			}
			{
				/*
				Subarray with given XOR
				https://www.interviewbit.com/problems/subarray-with-given-xor/

				#Good
				*/
				int Solution::solve(vector<int> &A, int B) {
					unordered_map<int, int> M;
					int x = 0, count = 0;
					M[0] = 1;
					for (int i = 0; i < A.size(); i++) {
						x = x ^ A[i];
						count += M[x ^ B];
						M[x]++;
					}
					return count;
				}
			}
			{
				/*
				Two out of Three
				https://www.interviewbit.com/problems/two-out-of-three/

				#Good
				*/
				//Best Solution
				{
					vector<int> Solution::solve(vector<int> &a, vector<int> &b, vector<int> &c) {
						vector<int> cnt(100010);
						for (auto x : a) cnt[x] |= 1;
						for (auto x : b) cnt[x] |= 2;
						for (auto x : c) cnt[x] |= 4;

						vector<int> ans;
						for (int i = 1; i <= 100000; i++) {
							if (cnt[i] == 3 || cnt[i] >= 5) ans.push_back(i);
						}
						return ans;
					}
				}
				// Good Solution
				{
					vector<int> Solution::solve(vector<int> &A, vector<int> &B, vector<int> &C) {
						int D[100001] = {};
						int E[100001] = {};
						vector<int> S;
						for (int a : A) {
							if (D[a] == 0)D[a]++;
						}
						for (int a : B) {
							if (E[a] == 0)E[a]++;
						}
						for (int i = 0; i < 100001; i++) {
							D[i] += E[i];
						}
						for (int a : C) {
							if (D[a] >= 1)D[a]++;
						}
						for (int i = 0; i < 100001; i++) {
							if (D[i] >= 2) {
								S.push_back(i);
							}
						}
						return S;
					}
				}
			}
			{
				/*
				Substring Concatenation
				https://www.interviewbit.com/problems/substring-concatenation/
				*/
				bool check(string S, int n, unordered_map<string, int> &M) {
					unordered_map<string, int> map;
					for (int i = 0; i < S.length() - n + 1; i += n) {
						string A = S.substr(i, n);
						if (M.find(A) == M.end())return false;
						map[A]++;
						if (map[A] > M[A]) {
							return false;
						}
					}
					return true;
				}
				vector<int> Solution::findSubstring(string A, const vector<string> &B) {
					int n = B[0].length(), m = n * B.size(); //n is length of a word in B, m is total length after concat
					vector<int> ans;
					if (m > A.length()) {
						return ans;
					}
					unordered_map<string, int> M;
					for (int i = 0; i < B.size(); i++) {
						M[B[i]]++;
					}

					for (int i = 0; i < A.length() - m + 1; i++) {
						if (check(A.substr(i, m), n, M)) {
							ans.push_back(i);
						}
					}
					return ans;
				}
			}
		}
		// Hashing Two Pointer
		{
			{
				/*
				Subarray with B odd numbers
				https://www.interviewbit.com/problems/subarray-with-b-odd-numbers/

				#Good #Revise
				*/
				int Solution::solve(vector<int> &A, int B) {
					unordered_map<int, int> map;
					int curr_sum = 0; //prefix sum
					int count = 0; //count of all subarrays

					for (int i = 0; i < A.size(); i++) {
						if (A[i] % 2 == 0) A[i] = 0;
						else A[i] = 1; // odd nos become 1. Now subarrays with sum = B should be found
					}

					for (int i = 0; i < A.size(); i++) {
						curr_sum += A[i];
						if (curr_sum == B) count += 1;
						if (map.find(curr_sum - B) != map.end()) {
							count += map[curr_sum - B];
						}
						map[curr_sum]++;
					}
					return count;
				}
			}
			{
				/*
				Minimum Window Substring
				https://www.interviewbit.com/problems/window-string/
				https://leetcode.com/problems/minimum-window-substring/

				#Good #Famous
				*/

				class Solution {
				public:
					string minWindow(string s, string t) {
						vector<int> map(128, 0);
						for (auto c : t) map[c]++;
						int remaining = t.size(), begin = 0, end = 0, d = INT_MAX, head = 0;
						while (end < s.size()) {
							int &countE = map[s[end]];
							if (countE > 0) remaining--; //in t
							countE--;
							end++;
							while (remaining == 0) { //valid
								if (end - begin < d)  {
									d = end - begin;
									head = begin;
								}
								int &countS = map[s[begin]];
								if (countS == 0) remaining++; //make it invalid
								countS++;
								begin++;
							}
						}
						return d == INT_MAX ? "" : s.substr(head, d);
					}
				};
			}
			{
				/*
				Longest Substring Without Repeat
				https://www.interviewbit.com/problems/longest-substring-without-repeat/

				#Good
				*/
				int Solution::lengthOfLongestSubstring(string A) {
					int n = A.size();
					vector<int> index(256, -1);
					int ans, start;
					ans = 0;
					start = -1;
					for (int i = 0; i < n; i++) {
						if (index[A[i]] > start) {
							start = index[A[i]];
						}
						ans = max(ans, i - start);
						index[A[i]] = i;
					}
					return ans;
				}
			}
		}
	}
// Backtracking
	{
		// Example
		{
			{
				/*
				Reverse Link List Recursion
				https://www.interviewbit.com/problems/reverse-link-list-recursion

				#Famous #Good
				*/
				ListNode* Solution::reverseList(ListNode * A) {
					if (!A || !A->next) return A;
					ListNode* next = A->next;
					A->next = NULL;
					ListNode* head = reverseList(next);
					next->next = A;
					return head;
				}
			}
			{
				/*
				Modular Expression
				https://www.interviewbit.com/problems/modular-expression/
				*/
				int Solution::Mod(int A, int B, int C) {
					if (A == 0) return 0;
					if (B == 0) return 1;
					if (B == 1) return (A % C + C) % C;
					int b = B / 2;
					long long temp = Mod(A, b, C);
					long long ans = (temp * temp) % C;
					if (B % 2 != 0) ans *= A;

					return ans % C;
				}
			}
		}
		//Subsets
		{
			{
				/*
				Subset
				https://www.interviewbit.com/problems/subset/
				*/
				void subset(vector<int> &A, vector<vector<int>> &ans, vector<int> temp, int index) {

					ans.push_back(temp);

					for (int i = index; i < A.size(); i++) {
						temp.push_back(A[i]);
						subset(A, ans, temp, i + 1);
						temp.pop_back();
					}
					return;
				}

				vector<vector<int> > Solution::subsets(vector<int> &A) {
					vector<vector<int>> ans;
					sort(A.begin(), A.end());
					vector<int> temp;
					subset(A, ans, temp, 0);
					return ans;
				}
			}
			{
				/*
				Subsets II
				https://www.interviewbit.com/problems/subsets-ii/
				*/
				void subset(vector<int> &A, vector<vector<int>> &ans, vector<int> temp, int index) {
					ans.push_back(temp);

					for (int i = index; i < A.size(); i++) {
						temp.push_back(A[i]);
						subset(A, ans, temp, i + 1);
						while (i < A.size() - 1 && A[i] == A[i + 1]) i++; // here is the difference
						temp.pop_back();
					}
					return;
				}

				vector<vector<int> > Solution::subsetsWithDup(vector<int> &A) {
					vector<vector<int>> ans;
					sort(A.begin(), A.end());
					vector<int> temp;
					subset(A, ans, temp, 0);
					return ans;
				}
			}
			{
				/*
				Combination Sum
				https://www.interviewbit.com/problems/combination-sum/

				#Good
				*/
				void put(const vector<int> &A, vector<int> &C, vector<vector<int>> &ans, int index, int B) {
					if (B == 0) {
						ans.push_back(C);
					} else if (B > 0) {
						for (int i = index; i < A.size(); i++) {
							C.push_back(A[i]);
							put(A, C, ans, i, B - A[i]);
							while (i < A.size() - 1 && A[i] == A[i + 1]) i++;
							C.pop_back();
						}
					}
				}
				vector<vector<int>> Solution::combinationSum(vector<int> &A, int B) {
					sort(A.begin(), A.end());
					vector<vector<int>> ans;
					vector<int> temp;
					put(A, temp, ans, 0, B);
					return ans;
				}
			}
			{
				/*
				Combination Sum II
				https://www.interviewbit.com/problems/combination-sum-ii/
				*/
				void put(const vector<int> &A, vector<int> &C, vector<vector<int>> &ans, int index, int B) {
					if (B == 0) {
						ans.push_back(C);
					} else if (B > 0) {
						for (int i = index; i < A.size(); i++) {
							C.push_back(A[i]);
							put(A, C, ans, i + 1, B - A[i]); //diffrence in i+1
							while (i < A.size() - 1 && A[i] == A[i + 1]) i++;
							C.pop_back();
						}
					}
				}
				vector<vector<int>> Solution::combinationSum(vector<int> &A, int B) {
					sort(A.begin(), A.end());
					vector<vector<int>> ans;
					vector<int> temp;
					put(A, temp, ans, 0, B);
					return ans;
				}
			}
			{
				/*
				Combinations
				https://www.interviewbit.com/problems/combinations/
				*/
				void comb(vector<vector<int>> &S, const int &A, int B, vector<int> &temp, int index) {
					if (B == 0) {
						S.push_back(temp);
						return;
					}
					for (int i = index; i <= A; i++) {
						temp.push_back(i);
						comb(S, A, B - 1, temp, i + 1);
						temp.pop_back();
					}
				}
				vector<vector<int> > Solution::combine(int A, int B) {
					vector<vector<int>> S;
					vector<int> temp;
					comb(S, A, B, temp, 1);
					return S;
				}
			}
		}
		//Maths and Backtracking
		{
			{
				/*
				Maximal String
				https://www.interviewbit.com/problems/maximal-string/
				*/
				void maxstring(string A, int b, string & temp, int index) {
					temp = max(temp, A);
					if (b == 0 || index == A.size()) return;

					maxstring(A, b, temp, index + 1);

					for (int i = index + 1; i < A.size(); i++) {
						if (A[index] < A[i]) {
							swap(A[index], A[i]);
							maxstring(A, b - 1, temp, index + 1);
							swap(A[i], A[index]);
						}
					}
				}

				string Solution::solve(string A, int B) {
					string temp = A;
					maxstring(A, B, temp, 0);
					return temp;
				}
			}
			{
				/*
				Gray Code
				https://www.interviewbit.com/problems/gray-code/

				#Good
				*/
				vector<int> Solution::grayCode(int A) {
					if (A == 1) {
						return {0, 1};
					}
					vector<int> S = grayCode(A - 1);
					int n = S.size();
					int x = 1 << (A - 1);
					for (int i = n - 1; i >= 0; i--) {
						S.push_back(x + S[i]);
					}
					return S;
				}
			}
			{
				/*
				Kth Permutation Sequence
				interviewbit.com/problems/kth-permutation-sequence/
				*/
				int fact(int x) {
					if (x > 13)return INT_MAX;
					if (x <= 1) {
						return 1;
					}
					return x * fact(x - 1);
				}
				string get(vector<string> &str, int B, int A) {
					if (A == 1) {
						return str[0];
					}
					int x = fact(A - 1);
					int i = 0;
					while (x * i < B) {
						i++;
					}
					i--;
					string ans = "";
					ans += str[i];
					str.erase(str.begin() + i);
					ans += get(str, B - i * x, A - 1);
					return ans;
				}
				string Solution::getPermutation(int A, int B) {
					vector<string> str;
					for (int i = 1; i <= A; i++) {
						str.push_back(to_string(i));
					}
					return get(str, B, A);
				}
			}
		}
		//Bruteforce Builder
		{
			{
				/*
				Letter Combinations of a Phone Number
				https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/?show=1
				https://www.interviewbit.com/problems/letter-phone/
				*/
				class Solution {
				private:
					vector<vector<char>> map = {
						{'a', 'b', 'c'},
						{'d', 'e', 'f'},
						{'g', 'h', 'i'},
						{'j', 'k', 'l'},
						{'m', 'n', 'o'},
						{'p', 'q', 'r', 's'},
						{'t', 'u', 'v'},
						{'w', 'x', 'y', 'z'}
					};
				public:
					vector<string> letterCombinations(string digits) {
						if (digits == "") return {};
						vector<string> ans = {""};
						return combi(0, digits, ans);
					}

					vector<string> combi(int i, string & digits, vector<string> &ans) {
						if (i >= digits.size()) return ans;
						vector<string> x = {};
						for (char &c : map[digits[i] - '2']) {
							for (string &w : ans) {
								x.push_back(w + c);
							}
						}
						return combi(i + 1, digits, x);
					}
				};
			}
			{
				/*
				Palindrome Partitioning
				https://www.interviewbit.com/problems/palindrome-partitioning/
				*/
				bool ispali(string S) {
					int l = 0, r = S.length() - 1;
					while (l <= r) {
						if (S[l] != S[r])return false;
						l++;
						r--;
					}
					return true;
				}
				void pali(vector<vector<string>> &S, vector<string> &temp, string A) {
					if (A.length() == 0) {
						S.push_back(temp);
						return;
					}
					for (int i = 1; i <= A.length(); i++) {
						if (ispali(A.substr(0, i))) {
							temp.push_back(A.substr(0, i)); //A.substr(0,i) => copy from 0 to i-1
							pali(S, temp, A.substr(i)); //A.substr(i)=> copy starting from i to end
							temp.pop_back();
						}
					}
				}
				vector<vector<string> > Solution::partition(string A) {
					vector<vector<string>> S;
					vector<string> temp;
					pali(S, temp, A);
					return S;
				}
			}
			{
				/*
				Generate all Parentheses II
				https://www.interviewbit.com/problems/generate-all-parentheses-ii/
				*/
				//Good Sol
				{
					void helper(vector<string> &S, string & temp, int open, int close) {
						if (open == 0 && close == 0) {
							S.push_back(temp);
						}
						if (open > 0) {
							temp.push_back('(');
							helper(S, temp, open - 1, close + 1);
							temp.pop_back();
						}
						if (close > 0) {
							temp.push_back(')');
							helper(S, temp, open, close - 1);
							temp.pop_back();
						}
					}
					vector<string> Solution::generateParenthesis(int A) {
						vector<string> S;
						string temp = "";
						helper(S, temp, A, 0);
						return S;
					}
				}
				//Good Sol
				{
					void helper(vector<string> &S, string & temp, int remaining, int current) {
						if (current > remaining) return;

						if (remaining == 0) {
							S.push_back(temp);
							return;
						}

						temp.push_back('(');
						helper(S, temp, remaining, current + 1);
						temp.pop_back();

						if (current == 0 ) return;
						temp.push_back(')');
						helper(S, temp, remaining - 1, current - 1);
						temp.pop_back();
					}

					vector<string> Solution::generateParenthesis(int A) {
						vector<string> S;
						string temp = "";
						helper(S, temp, A, 0);
						return S;
					}
				}
			}
		}
		//Permutations
		{
			{
				/*
				Permutations
				https://www.interviewbit.com/problems/permutations/

				#Famous
				*/
				void permute2(vector<int> &num, int start, vector<vector<int> > &result) {
					if (start == num.size() - 1) {
						result.push_back(num);
						return;
					}
					for (int i = start; i < num.size(); i++) {
						swap(num[start], num[i]);
						permute2(num, start + 1, result);
						swap(num[start], num[i]);
					}
				}

				vector<vector<int> > Solution::permute(vector<int> &num) {
					vector<vector<int> > result;
					if (num.size() == 0)
						return result;
					sort(num.begin(), num.end());
					permute2(num, 0, result);
					return result;
				}
			}
		}
		//Game Solving
		{
			{
				/*
				NQueens
				https://www.interviewbit.com/problems/nqueens/
				*/
				vector<vector<bool>> fill(vector<vector<bool>> X, int i, int j) {
					int n = X.size();
					for (int a = 0; a < n; a++) {
						X[i][a] = 0;
						X[a][j] = 0;
					}
					for (int a = i + 1, b = j + 1; a < n && b < n; a++, b++) {
						X[a][b] = 0;
					}
					for (int a = i + 1, b = j - 1; a < n && b >= 0; a++, b--) {
						X[a][b] = 0;
					}
					return X;
				}
				void push(int n, vector<vector<string>> &S, vector<int> &Z) {
					vector<string> SS(n, string(n, '.'));
					for (int i = 0; i < n; i++) {
						SS[i][Z[i]] = 'Q';
					}
					S.push_back(SS);
				}
				void check(vector<vector<bool>> X, int i,  vector<vector<string>> &S, vector<int> &Z) {
					int n = X.size();
					for (int a = 0; a < n; a++) {
						if (X[i][a]) {
							Z[i] = a;
							if (i + 1 == n) {
								push(X.size(), S, Z);
							} else {
								check(fill(X, i, a), i + 1, S, Z);
							}
						}
					}
				}
				vector<vector<string> > Solution::solveNQueens(int A) {
					vector<vector<string>> S;
					vector<int> Z(A, 0);
					vector<vector<bool>> X(A, vector<bool> (A, 1));
					check(X, 0, S, Z);
					return S;
				}
			}
			{
				/*
				N-Queens II
				https://leetcode.com/problems/n-queens-ii/description/
				*/
				class Solution {
				public:
					int totalNQueens(int n) {
						int ans;
						vector<vector<bool>> check = { vector<bool>(n), vector<bool>(2 * n - 1), vector<bool>(2 * n - 1)}; // V, DU, DD
						fun(check, n, ans, 0, 0);
						return ans;
					}

					void fun(vector<vector<bool>> &check, int &n, int &ans, int r, int c) {
						if (r == n) {
							ans++;
							return;
						}
						for (int j = 0; j < n; j++) {
							if (!isValid(check, r, j, n)) continue;

							setCheck(r, j, check, true, n);
							fun(check, n, ans, r + 1, j + 1);
							setCheck(r, j, check, false, n);
						}
					}

					void setCheck(int &i, int &j, vector<vector<bool>> &check, bool val, int &n) {
						check[0][j] = val;
						check[1][i + j] = val;
						check[2][j - i + n - 1] = val;
					}

					bool isValid(vector<vector<bool>> &check, int i, int j, int &n) {
						return !(check[0][j] || check[1][i + j] || check[2][j - i + n - 1]);
					}

				};
			}
			{
				/**/
				// My Sol
				{
					bool check(vector<vector<char>> &A, int x, char a) {
						int i = x / 9, j = x % 9;
						for (int b = 0; b < 9; b++) {
							if (A[b][j] == a) {
								return false;
							}
							if (A[i][b] == a) {
								return false;
							}
						}
						for (int b = 0 + (3 * (i / 3)); b < 3 + (3 * (i / 3)); b++) {
							for (int c = 0 + (3 * (j / 3)); c < 3 + (3 * (j / 3)); c++) {
								if (A[b][c] == a) {
									return false;
								}
							}
						}
						return true;
					}
					bool fill(int x, vector<vector<char>> &B) {
						for ( ; x < 81; x++) {
							if (B[x / 9][x % 9] == '.') {
								break;
							}
						}
						if (x == 81) {
							return true;
						}
						for (char a = '1'; a < '9' + 1; a++) {
							if (check(B, x, a)) {
								B[x / 9][x % 9] = a;
								if (fill(x + 1, B)) {
									return true;
								}
							}
						}
						B[x / 9][x % 9] = '.';
						return false;
					}
					void Solution::solveSudoku(vector<vector<char> > &A) {
						fill(0, A);
					}
				}
			}
		}
	}
// Stacks and Queues
	{
		// Example
		{
			/*
			Generate all Parentheses
			https://www.interviewbit.com/problems/generate-all-parentheses/

			#Famous
			*/
			int Solution::isValid(string A) {
				stack<char> s;
				for (char c : A) {
					if (c == '(' || c == '{' || c == '[') {
						s.push(c);
					} else {
						if (s.empty()) return 0; // There is no opening bracket to match
						char top = s.top();
						s.pop();
						if ((c == ')' && top != '(') || (c == '}' && top != '{') || (c == ']' && top != '[')) {
							return 0; // Mismatched brackets
						}
					}
				}
				return s.empty(); // If stack is empty, all brackets matched
			}
		}
		// Stack simple
		{
			{
				/*
				Balanced Parantheses!
				https://www.interviewbit.com/problems/balanced-parantheses/
				*/
				int Solution::solve(string A) {
					stack<char> s;
					for (char c : A) {
						if (c == '(') {
							s.push(c);
						} else if (c == ')') {
							if (s.empty() || s.top() != '(') {
								return 0; // Unmatched closing parenthesis
							}
							s.pop();
						}
					}
					return s.empty(); // If stack is empty, all parentheses matched
				}
			}
			{
				/*
				Simplify Directory Path
				https://www.interviewbit.com/problems/simplify-directory-path/
				*/
				{
					string Solution::simplifyPath(string A) {
						string S = "/";
						int n = A.length();
						int i = 0;
						while (i < n - 1) {
							if (A[i] == '/') {
								while (A[i] == '/' && i < n)i++;
							} else if (A[i] == '.' && A[i + 1] == '/') {
								i += 1;
								continue;
							} else if (A[i] == '.' && A[i + 1] == '.') {
								i += 2;
								if (S.length() == 1)continue;
								S.pop_back();
								while (S.back() != '/') {
									S.pop_back();
								}
							}
							else {
								while (A[i] != '/' && i < n) {
									S += A[i];
									i++;
								}
								S += '/';
							}
						}
						if (S.length() != 1)S.pop_back();
						return S;
					}
				}
				{
					/*
					Redundant Braces
					https://www.interviewbit.com/problems/redundant-braces/
					*/
					int Solution::braces(string A) {
						int n = A.size();
						stack<char> s;
						for (int i = 0; i < n; i++) {
							if (A[i] == '(' || A[i] == '+' || A[i] == '-' || A[i] == '*' || A[i] == '/') {
								s.push(A[i]);
							}
							if (A[i] == ')') {
								if (s.top() == '(') {
									return 1;
								}
								else {
									while (s.top() == '+' || s.top() == '-' || s.top() == '*' || s.top() == '/') {
										s.pop();
									}
									s.pop();
								}
							}
						}
						return 0;
					}
				}
				{
					/*
					Min Stack
					https://www.interviewbit.com/problems/min-stack/

					#Good
					*/
					{
						stack<int> A;
						stack<int> mins;
						MinStack::MinStack() {
							while (!A.empty()) {
								A.pop();
							}
							while (!mins.empty()) {
								mins.pop();
							}
						}

						void MinStack::push(int x) {
							if (A.empty()) {
								mins.push(x);
							}
							else if (mins.top() < x) {
								mins.push(mins.top());
							} else {
								mins.push(x);
							}
							A.push(x);
						}
						void MinStack::pop() {
							if (A.empty()) {
								return;
							}
							A.pop();
							mins.pop();
						}

						int MinStack::top() {
							if (A.empty()) {
								return -1;
							}
							return A.top();
						}

						int MinStack::getMin() {
							if (A.empty()) {
								return -1;
							}
							return mins.top();
						}
					}
				}
			}
		}
		//Clever Stack
		{
			/*
			MAXSPPROD
			https://www.interviewbit.com/problems/maxspprod/

			#Good
			*/
			int Solution::maxSpecialProduct(vector<int> &A) {
				int n = A.size();
				vector<int> LeftSpecialValue(n, 0), RightSpecialValue(n, 0);
				stack<int> leftCalc;
				leftCalc.push(0);
				LeftSpecialValue[0] = 0;
				for (int i = 1; i < n; i++) {
					while (!leftCalc.empty() && A[leftCalc.top()] <= A[i]) {
						leftCalc.pop();
					}
					LeftSpecialValue[i] = (leftCalc.empty()) ? 0 : leftCalc.top();
					leftCalc.push(i);
				}
				stack<int> rightCalc;
				rightCalc.push(n - 1);
				RightSpecialValue[n - 1] = 0;
				for (int i = n - 2; i >= 0; i--) {
					while (!rightCalc.empty() && A[rightCalc.top()] <= A[i]) {
						rightCalc.pop();
					}
					RightSpecialValue[i] = (rightCalc.empty()) ? 0 : rightCalc.top();
					rightCalc.push(i);
				}
				long long mx = -1;
				for (int i = 0; i < n; i++) {
					mx = max(mx, 1LL * LeftSpecialValue[i] * RightSpecialValue[i]);
				}
				return mx % 1000000007;
			}
		}
		{
			/*
			Nearest Smaller Element
			https://www.interviewbit.com/problems/nearest-smaller-element/

			#Good
			*/
			vector<int> Solution::prevSmaller(vector<int> &A) {
				stack<int> S;
				vector<int> ans(A.size(), -1);
				for (int i = 0; i < A.size(); i++) {
					while (!S.empty() && S.top() >= A[i]) {
						S.pop();
					}
					ans[i] = S.empty() ? -1 : S.top();
					S.push(A[i]);
				}
				return ans;
			}
		}
		{
			/*
			Largest Rectangle in Histogram
			https://www.interviewbit.com/problems/largest-rectangle-in-histogram/

			#Famous #Revise #Good
			*/

			//Understandable
			{
				int Solution::largestRectangleArea(vector<int> &A) {
					int n = A.size(), ans = 0;
					stack<int> S;
					vector<int> left(n, 0), right(n, n - 1);
					for (int i = 0; i < n; i++) {
						while (!S.empty() && A[S.top()] >= A[i]) {
							S.pop();
						}
						if (!S.empty())
							left[i] = S.top() + 1;
						S.push(i);
					}
					while (!S.empty())S.pop();
					for (int i = n - 1; i >= 0; i--) {
						while (!S.empty() && A[S.top()] >= A[i]) {
							S.pop();
						}
						if (!S.empty())
							right[i] = S.top() - 1;
						S.push(i);
					}
					for (int i = 0; i < n; i++) {
						ans = max(ans, A[i] * (right[i] - left[i] + 1));
					}
					return ans;
				}
			}
			//Best Sol
			{
				int Solution::largestRectangleArea(vector<int> &A) {
					stack<int> S;
					A.push_back(0);
					int sum = 0;
					for (int i = 0; i < A.size(); i++) {
						if (S.empty() || A[i] >= A[S.top()]) S.push(i);
						else {
							int tmp = S.top();
							S.pop();
							sum = max(sum, A[tmp] * (S.empty() ? i : i - S.top() - 1));
							i--;
						}
					}
					return sum;
				}
			}
		}
		{
			/*
			First non-repeating character in a stream of characters
			https://www.interviewbit.com/problems/first-non-repeating-character-in-a-stream-of-characters/

			#Good #Famous
			*/
			string Solution::solve(string A) {
				int count[26] = {};
				queue<char> Q;
				string S;
				for (char a : A) {
					count[a - 'a']++;
					Q.push(a);
					while (!Q.empty() && count[Q.front() - 'a'] > 1) {
						Q.pop();
					}
					if (Q.empty()) {
						S += '#';
					} else {
						S += Q.front();
					}
				}
				return S;
			}
		}
		{
			/*
			Sliding Window Maximum
			https://www.interviewbit.com/problems/sliding-window-maximum/
			https://leetcode.com/problems/sliding-window-maximum/description/

			#Famous #Good #Revise
			*/

			//Using Deque (Best)
			{
				vector<int> Solution::slidingMaximum(const vector<int> &A, int B) {
					int n = A.size();
					deque<int> S;
					vector<int> ans(n - B + 1, 0);
					for (int i = 0; i < B - 1; i++) {
						while (!S.empty() && A[S.back()] <= A[i]) {
							S.pop_back();
						}
						S.push_back(i);
					}
					for (int i = B - 1; i < n; i++) {
						while (!S.empty() && S.front() <= i - B) {
							S.pop_front();
						}
						while (!S.empty() && A[S.back()] <= A[i]) {
							S.pop_back();
						}
						S.push_back(i);
						ans[i - B + 1] = A[S.front()];
					}
					return ans;
				}
			}
		}
		{
			/*
			Evaluate Expression: Reverse Polish Notation
			https://www.interviewbit.com/problems/evaluate-expression/

			#Famous #Good
			*/
			int Solution::evalRPN(vector<string> &A) {
				stack<int> num;
				for (auto i : A) {
					if (i == "+" || i == "-" || i == "*" || i == "/") {
						int num1 = num.top(); num.pop();
						int num2 = num.top(); num.pop();
						if (i == "+") {
							num2 += num1;
						} else if (i == "-") {
							num2 -= num1;
						} else if (i == "*") {
							num2 *= num1;
						} else if (i == "/") {
							num2 /= num1;
						}
						num.push(num2);
					} else {
						num.push(stoi(i));
					}
				}
				return num.top();
			}
		}
		{
			/*
			Rain Water Trapped
			https://www.interviewbit.com/problems/rain-water-trapped/

			#Famous #Good #Revise
			*/
			// Good Sol
			{
				int Solution::trap(const vector<int> &arr) {
					int n = arr.size();
					int left[n];
					int right[n];
					left[0] = arr[0];
					right[n - 1] = arr[n - 1];

					for (int i = 1; i < n; i++) {
						left[i] = max(left[i - 1], arr[i]);
						right[n - i - 1] = max(right[n - i], arr[n - i - 1]);
					}

					int ans = 0;
					for (int i = 0; i < n; i++) {
						ans = ans + (min(left[i], right[i]) - arr[i]);
					}
					return ans;
				}
			}
			//Two pointer Best Solution
			{
				int Solution::trap(const vector<int> &A) {
					int len = A.size();
					int min = 0, prev = 0, i = 0, j = len - 1, ans = 0;
					while (i <= j) {
						min = A[i] <= A[j] ? i++ : j--;
						if (prev > A[min]) {
							ans += prev - A[min];
						}
						else
							prev = A[min];
					}
					return ans;
				}
			}
		}
	}
// Linked List
	{
		// List Sort
		{
			{
				/*
				Sort Binary Linked List
				https://www.interviewbit.com/problems/sort-binary-linked-list/
				*/
				ListNode* Solution::solve(ListNode * A) {
					ListNode *zero = new ListNode(0), *one = new ListNode(0);
					ListNode *zero2 = zero, *one2 = one;
					while (A != NULL) {
						if (A->val == 0) {
							zero->next = A;
							zero = zero->next;
						} else {
							one->next = A;
							one = one->next;
						}
						A = A->next;
					}
					one->next = NULL;
					zero->next = one2->next;
					return zero2->next;
				}
			}
			{
				/*
				Partition List
				https://www.interviewbit.com/problems/partition-list/
				*/
				ListNode* Solution::partition(ListNode * A, int B) {
					ListNode *small = new ListNode(0), *large = new ListNode(0);
					ListNode *small2 = small, *large2 = large;
					while (A != NULL) {
						if (A->val < B) {
							small->next = A;
							small = small->next;
						} else {
							large->next = A;
							large = large->next;
						}
						A = A->next;
					}
					large->next = NULL;
					small->next = large2->next;
					return small2->next;
				}
			}
			{
				/*
				Sort List
				https://www.interviewbit.com/problems/sort-list/
				*/
				ListNode * merge(ListNode * A, ListNode * B) {
					ListNode* head = new ListNode(0), *ans = head;
					while (A && B) {
						if ((A->val) > (B->val)) {
							head->next = B;
							B = B->next;
						} else {
							head->next = A;
							A = A->next;
						}
						head = head->next;
					}
					if (A == NULL) {
						A = B;
					}
					while (A) {
						head->next = A;
						A = A->next;
						head = head->next;
					}
					return ans->next;
				}
				ListNode * sort(ListNode * start, int n) {
					if (n == 0)return NULL;
					else if (n == 1) {
						start->next = NULL;
						return start;
					}
					else if (n == 2) {
						if ((start->val) > (start->next->val)) {
							ListNode* temp = start->next;
							temp->next = start;
							start->next = NULL;
							return temp;
						}
						start->next->next = NULL;
						return start;
					} else {
						ListNode *M = start;
						int i = 1;
						while (i <= n / 2) {
							M = M->next;
							i++;
						}
						return merge(sort(start, n / 2), sort(M, n - n / 2));
					}
				}
				ListNode* Solution::sortList(ListNode * A) {
					ListNode *slow = A, *fast = A;
					int n = 1;
					while (fast->next && fast->next->next) {
						fast = fast->next->next;
						slow = slow->next;
						n++;
					}
					if (fast->next) {
						fast = sort(slow->next, n);
					} else {
						fast = sort(slow->next, n - 1);
					}
					return merge(sort(A, n), fast);
				}
			}
		}
		// List 2 Pointer
		{
			{
				/*
				Palindrome List
				https://www.interviewbit.com/problems/palindrome-list/

				#Good
				*/
				ListNode * reverse(ListNode * head) {
					if (head == nullptr || head->next == nullptr) return head;
					ListNode *prev = head, *curr = head->next, *next = nullptr; prev->next = nullptr;
					while (curr != nullptr) {
						next = curr->next;
						curr->next = prev;
						prev = curr;
						curr = next;
					}
					return prev;
				}
				int Solution::lPalin(ListNode * head) {
					if (head == nullptr) return 1;
					int count = 0;
					ListNode* ptr = head;
					while (ptr != nullptr) {ptr = ptr->next; count++;}
					ListNode *firstHalf = nullptr, *secondHalf = nullptr;
					ptr = head;
					count = (count + 1) / 2;
					int start = 0;
					while (ptr != nullptr && start < count - 1) {
						ptr = ptr->next; start++;
					}
					firstHalf = head;
					secondHalf = ptr->next; ptr->next = nullptr;
					secondHalf = reverse(secondHalf);
					ListNode *ptr1 = firstHalf, *ptr2 = secondHalf;
					while (ptr1 != nullptr && ptr2 != nullptr) {
						if (ptr1->val != ptr2->val) return 0;
						ptr1 = ptr1->next; ptr2 = ptr2->next;
					}
					return 1;
				}
			}
			{
				/*
				Remove Duplicates from Sorted List II
				https://www.interviewbit.com/problems/remove-duplicates-from-sorted-list-ii/
				*/
				ListNode* Solution::deleteDuplicates(ListNode * A) {
					ListNode* dummy = new ListNode(0), *p1 = dummy;
					dummy->next = A;

					while (p1->next and p1->next->next) {
						if (p1->next->val == p1->next->next->val) {
							int k = p1->next->val;
							while (p1->next and p1->next->val == k)
								p1->next = p1->next->next;
						}
						else  p1 = p1->next;
					}
					return dummy->next;
				}
			}
			{
				/*
				Merge Two Sorted Lists
				https://www.interviewbit.com/problems/merge-two-sorted-lists/
				*/
				ListNode* Solution::mergeTwoLists(ListNode * A, ListNode * B) {
					ListNode *head = new ListNode(0), *temp = head;
					while (A && B) {
						if (A->val < B->val) {
							temp->next = A;
							A = A->next;
						} else {
							temp->next = B;
							B = B->next;
						}
						temp = temp->next;
					}
					(A == NULL) ? (A = B) : 0;
					temp->next = A;
					return head->next;
				}
			}
			{
				/*
				Remove Duplicates from Sorted List
				https://www.interviewbit.com/problems/remove-duplicates-from-sorted-list/
				*/
				ListNode* Solution::deleteDuplicates(ListNode * A) {
					ListNode *head = new ListNode(INT_MIN), *ans = head;
					while (A) {
						if (A->val != head->val) {
							head = head->next = A;
						}
						A = A->next;
					}
					head->next = NULL;
					return ans->next;
				}
			}
			{
				/*
				Remove Nth Node from List End
				https://www.interviewbit.com/problems/remove-nth-node-from-list-end/

				#Good
				*/
				ListNode* Solution::removeNthFromEnd(ListNode * A, int B) {
					ListNode *curr = A, *prev = A;
					while (B-- > 0 && curr) {
						curr = curr->next;
					}
					if (!curr) {
						return A->next;
					}
					while (curr->next) {
						curr = curr->next;
						prev = prev->next;
					}
					prev->next = prev->next->next;
					return A;
				}
			}
		}
		// Pointer Move
		{
			{
				/*
				K reverse linked list
				https://www.interviewbit.com/problems/k-reverse-linked-list/

				#Famous
				*/
				ListNode* Solution::reverseList(ListNode * A, int B) {
					ListNode*cur = A, *prev = NULL, *nex;
					int c = 0;
					while (cur && c < B) {
						nex = cur->next;
						cur->next = prev;
						prev = cur;
						cur = nex;
						c++;
					}
					if (nex != NULL)A->next = reverseList(nex, B);
					return prev;
				}
			}
			{
				/*
				Even Reverse
				https://www.interviewbit.com/problems/even-reverse/
				*/
				// Best Sol
				{
					ListNode* Solution::solve(ListNode * A) {
						if (A->next == NULL || A->next->next == NULL || A->next->next->next == NULL) return A;
						vector<ListNode*>v;
						ListNode* temp = A;
						int i = 0;
						while (A != NULL) {
							if (i % 2 == 1) v.push_back(A);
							A = A->next;
							i++;
						}
						int l;
						for (int i = 0; i < v.size() / 2; i++) {
							swap(v[i]->val, v[v.size() - 1 - i]->val);
						}
						return temp;
					}
				}
				// Solution
				{
					ListNode * reverse(ListNode * A) {
						ListNode *prev = NULL;
						while (A) {
							ListNode *next = A->next;
							A->next = prev;
							prev = A;
							A = next;
						}
						return prev;
					}
					ListNode* Solution::solve(ListNode * A) {
						ListNode *odd = new ListNode(0), *even = new ListNode(0);
						ListNode *hodd = odd, *heven = even;
						while (A && A->next) {
							odd->next = A;
							even->next = A->next;
							odd = A;
							even = A->next;
							A = A->next->next;
						}
						odd->next = A;
						even->next = NULL;
						even = reverse(heven->next);
						odd = hodd->next;
						while (even) {
							ListNode *temp = odd->next;
							odd->next = even;
							even = even->next;
							odd->next->next = temp;
							odd = temp;
						}
						return hodd->next;
					}
				}
			}
			{
				/*
				Swap List Nodes in pairs
				https://www.interviewbit.com/problems/swap-list-nodes-in-pairs/
				*/
				ListNode* Solution::swapPairs(ListNode * A) {
					if (A->next == NULL) {
						return A;
					}
					ListNode* temp = new ListNode(0), *ans = temp;
					while (A && A->next) {
						ListNode* temp2 = A->next->next;
						temp->next = A->next;
						A->next->next = A;
						A->next = temp2;
						temp = A;
						A = temp2;
					}
					return ans->next;
				}
			}
			{
				/*
				Rotate List
				https://interviewbit.com/problems/rotate-list/

				#Famous
				*/
				ListNode* Solution::rotateRight(ListNode * A, int B) {
					ListNode *head = new ListNode(0), *C = head;
					head->next = A;
					int n = 1;
					while (A && A->next) {
						n++;
						A = A->next;
					}
					A->next = head->next;
					B = B % n;
					A = head->next;
					while (n - B) {
						head = head->next;
						n--;
					}
					C->next = head->next;
					head->next = NULL;
					return C->next;
				}
			}
		}
		// List Trick
		{
			{
				/*
				Reverse Link List II
				https://www.interviewbit.com/problems/reverse-link-list-ii/
				*/
				ListNode * reverseBetween(ListNode * head, int left, int right) {
					ListNode *curr = new ListNode(0, head), *prev, *next;
					int i = 0;
					while (curr != NULL && i < left) {
						prev = curr;
						curr = curr->next;
						i++;
					}
					ListNode *leftStart = prev;

					while (curr != NULL && i <= right) {
						next = curr->next;
						curr->next = prev;

						prev = curr;
						curr = next;
						i++;
					}
					if (leftStart != NULL && leftStart->next != NULL) {
						leftStart->next->next = curr;
						leftStart->next = prev;
					}
					if (left == 1) return leftStart->next;
					return head;
				}
			}
			{
				/*
				Reorder List
				https://www.interviewbit.com/problems/reorder-list/

				#Good
				*/
				ListNode * reverse(ListNode * A) {
					ListNode *prev = NULL, *next;
					while (A) {
						next = A->next;
						A->next = prev;
						prev = A;
						A = next;
					}
					return prev;
				}
				ListNode* Solution::reorderList(ListNode * A) {
					ListNode *slow = A, *fast = A, *ans = A, *B;
					while (fast->next && fast->next->next) {
						slow = slow->next;
						fast = fast->next->next;
					}
					B = reverse(slow->next);
					slow->next = NULL;

					ListNode *C = A, *D = B;
					while (B) {
						C = A->next;
						D = B->next;
						A->next = B;
						B->next = C;
						B = D;
						A = C;
					}
					C = ans;
					return ans;
				}
			}
		}
		// List Math
		{
			{
				/*
				List Cycle
				https://www.interviewbit.com/problems/list-cycle/

				#Famous #Revise #FloydsCycleDetection #Good
				*/
				ListNode* Solution::detectCycle(ListNode * head) {
					if (head == NULL || head->next == NULL) return NULL;

					ListNode* slow = head;
					ListNode* fast = head;
					bool isCycle = false;

					while (slow != NULL && fast != NULL) {
						slow = slow->next;
						if (fast->next == NULL) return NULL;
						fast = fast->next->next;
						if (slow == fast) { isCycle = true; break; }
					}

					if (!isCycle) return NULL;
					slow = head;
					while ( slow != fast) {
						slow = slow->next;
						fast = fast->next;
					}

					return slow;
				}
			}
		}
	}
// Two Pointers
	{
		// Sorting
		{
			{
				/*
				Pair With Given Difference
				https://www.interviewbit.com/problems/pair-with-given-difference/

				#Revise
				*/
				// Hashing
				{
					int Solution::solve(vector<int> &A, int B) {
						unordered_set<int> M;
						for (int i = 0; i < A.size(); i++) {
							if (M.find(B + A[i]) != M.end() || M.find(A[i] - B) != M.end()) return 1;

							if (M.find(A[i]) == M.end()) M.insert(A[i]);
						}
						return 0;
					}
				}
				// Two Pointer
				{
					/*
					Pair With Given Difference
					https://www.interviewbit.com/problems/pair-with-given-difference/
					*/
					int Solution::solve(vector<int> &A, int B) {
						int n = A.size();
						sort(A.begin(), A.end());
						int left = 0, right = 1; // Because here we are checking for (A[i] - A[j])
						B = abs(B);

						while (right < n) {
							int diff = A[right] - A[left];
							if (diff == B && left != right) {
								return 1; // Found a pair with difference B
							} else if (diff < B) {
								right++; // Increase the right pointer
							} else {
								left++; // Increase the left pointer
							}
						}

						return 0; // No pair found with difference B
					}
				}
				{
					/*
					3 Sum
					https://www.interviewbit.com/problems/3-sum/

					#Revise
					*/
					int Solution::threeSumClosest(vector<int> &A, int B) {
						sort(A.begin(), A.end());
						int closest = A[0] + A[1] + A[2];
						for (int left = 0; left < A.size() - 2; left++) {
							int right = A.size() - 1, mid = left + 1; // Because here we are checking for (A[i] + A[j])
							while (mid < right) {
								int temp = A[left] + A[mid] + A[right];
								if (abs(temp - B) < abs(closest - B)) {
									closest = temp;
								}
								if (temp < B) {
									mid++;
								} else if (temp > B) {
									right--;
								} else {
									return B;
								}
							}
						}
						return closest;
					}
				}
				{
					/*
					Counting Triangles
					https://www.interviewbit.com/problems/counting-triangles/

					#Good
					*/
					int Solution::nTriang(vector<int> &A) {
						sort(A.begin(), A.end());
						int n = A.size();
						long long cnt = 0;
						for (int i = n - 1; i >= 2; i--) {
							int l = 0, r = i - 1;
							while (l < r) {
								if (A[l] + A[r] > A[i]) {
									cnt = cnt + r - l;
									cnt = cnt % 1000000007;
									r--;
								}
								else l++;
							}
						}
						return cnt;
					}
				}
				{
					/*
					Diffk
					https://www.interviewbit.com/problems/diffk/

					#Revise
					*/
					int Solution::diffPossible(vector<int> &A, int B) {
						int l = 0, r = 1;
						while (r < A.size()) {
							if (A[r] - A[l] < B) {
								r++;
							} else if (A[r] - A[l] > B) {
								l++;
							} else {
								if (l == r) r++;
								else return 1;
							}
						}
						return 0;
					}
				}
			}
			// Tricks
			{
				{
					/*
					Maximum Ones After Modification
					https://www.interviewbit.com/problems/maximum-ones-after-modification/

					#Good
					*/
					int Solution::solve(vector<int> &A, int B) {
						int l = -1, r = 0, length = 0, zeros = 0;
						while (r < A.size()) {
							if (A[r] == 1) {
								length = max(length, r - l);
							} else if (A[r] == 0) {
								zeros++;
								if (zeros <= B) {
									length = max(length, r - l);
								} else {
									l++;
									while (l <= r && A[l] == 1) {
										l++;
									}
									zeros--;
								}
							}
							r++;
						}
						return length;
					}
				}
				{
					/*
					Counting Subarrays! You need to find the number of subarrays in A having sum less than B
					https://www.interviewbit.com/problems/counting-subarrays/
					*/
					// Easy to understand
					{
						int Solution::solve(vector<int> &A, int B) {
							int n = A.size(), count = 0, sum = 0, j = 0;

							for (int i = 0; i < n; i++) {
								sum += A[i];
								while (sum >= B) {
									sum -= A[j];
									j++;
								}
								count += i - j + 1;
							}
							return count;
						}
					}
					// Good
					{
						int Solution::solve(vector<int> &A, int B) {
							int l = -1, r = -1, count = 0, sum = 0, n = A.size();
							while (r < n) {
								if (sum < B) {
									count += r - l;
									r++;
									sum += A[r];
								} else {
									l++;
									sum -= A[l];
								}
							}
							return count;
						}
					}
				}
				{
					/*
					Subarrays with distinct integers!
					https://www.interviewbit.com/problems/subarrays-with-distinct-integers/

					#Good
					*/
					int helper(vector<int> &a, int b) {
						int res = 0;
						int left = 0, right = 0;
						unordered_map<int, int> m;
						while (right < a.size()) {
							m[a[right]]++;
							while (m.size() > b) {
								m[a[left]]--;
								if (m[a[left]] == 0) m.erase(a[left]);
								left++;
							}
							res += right - left + 1;
							right++;
						}
						return res;
					}

					int Solution::solve(vector<int> &a, int b) {
						return helper(a, b) - helper(a, b - 1);
					}
				}
				{
					/*
					Array 3 Pointers
					https://www.interviewbit.com/problems/array-3-pointers/

					#Good
					*/
					int Solution::minimize(const vector<int> &A, const vector<int> &B, const vector<int> &C) {
						int i = 0, j = 0, k = 0;
						int temp, mini = INT_MAX;
						while (i < A.size() && j < B.size() && k < C.size()) {
							temp = max({abs(A[i] - B[j]), abs(A[i] - C[k]), abs(B[j] - C[k])});
							mini = min(temp, mini);

							if (A[i] <= B[j] && A[i] <= C[k]) {
								i++;
							} else if (A[i] >= B[j] && B[j] <= C[k]) {
								j++;
							} else if (A[i] >= C[k] && B[j] >= C[k]) {
								k++;
							}
						}
						return mini;
					}
				}
				{
					/*
					Container With Most Water
					https://www.interviewbit.com/problems/container-with-most-water/

					#Famous #Good #Revise
					*/
					int Solution::maxArea(vector<int> &A) {
						int n = A.size();
						int maxarea = 0, area;
						int l = 0, r = n - 1;
						while (l < r) {
							area = (r - l) * min(A[l], A[r]);
							maxarea = max(area, maxarea);
							if (A[l] > A[r]) {
								r--;
							} else {
								l++;
							}
						}
						return maxarea;
					}
				}
			}
			// Multiple Arrays
			{
				{
					/*
					Merge Two Sorted Lists II
					https://www.interviewbit.com/problems/merge-two-sorted-lists-ii/

					#Good
					*/
					void Solution::merge(vector<int> &A, vector<int> &B) {
						int a = A.size() - 1, b = B.size() - 1;
						A.resize(A.size() + B.size());
						int i = a + b + 1;
						while (b >= 0) {
							if (a == -1 || B[b] > A[a]) {
								A[i] = B[b];
								i--;
								b--;
							} else {
								A[i] = A[a];
								i--;
								a--;
							}
						}
						return ;
					}
				}
				{
					/*
					Intersection Of Sorted Arrays
					https://www.interviewbit.com/problems/intersection-of-sorted-arrays/
					*/
					vector<int> Solution::intersect(const vector<int> &A, const vector<int> &B) {
						int a = 0, b = 0;
						vector<int> S;
						while (a < A.size() && b < B.size()) {
							if (A[a] < B[b]) {
								a++;
							} else if (A[a] > B[b]) {
								b++;
							} else {
								S.push_back(A[a]);
								a++;
								b++;
							}
						}
						return S;
					}
				}
			}
			// Inplace Update
			{
				{
					/*
					Remove Duplicates from Sorted Array
					https://www.interviewbit.com/problems/remove-duplicates-from-sorted-array/

					#Famous
					*/
					int Solution::removeDuplicates(vector<int> &A) {
						int j = 1;
						for (int i = 1; i < A.size(); i++) {
							if (A[i] != A[i - 1]) {
								A[j] = A[i];
								j++;
							}
						}
						return j;
					}
				}
				{
					/*
					Remove Duplicates from Sorted Array II
					https://www.interviewbit.com/problems/remove-duplicates-from-sorted-array-ii/

					#Famous #Good
					*/
					int Solution::removeDuplicates(vector<int> &A) {
						int count = 0, n = A.size();
						for (int i = 0; i < n; i++) {
							if (i < n - 2 && A[i] == A[i + 1] && A[i] == A[i + 2]) continue;
							else {
								A[count] = A[i];
								count++;
							}
						}
						return count;
					}
				}
				{
					/*
					Sort by Color
					https://www.interviewbit.com/problems/sort-by-color/

					#Good #Revise
					*/
					void Solution::sortColors(vector<int> &A) {
						int i = -1, k = A.size();
						for (int j = 0; j < A.size() && j < k; j++) {
							if (A[j] < 1) {
								i++;
								swap(A[i], A[j]);
							}
							else if (A[j] > 1) {
								k--;
								swap(A[k], A[j]);
								j--;
							}
						}
					}
				}
			}
		}
// Binary Search
		/*
		https://leetcode.com/discuss/study-guide/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems
		*/
		{
			//Simple Binary Search
			{
				{
					/*
					Search in Bitonic Array!
					https://www.interviewbit.com/problems/search-in-bitonic-array/
					*/
					int Solution::solve(vector<int> &A, int B) {
						int start = 1, n = A.size(), end = n - 1, mid;
						while (start <= end) {
							mid = start + (end - start) / 2;
							if (A[mid] > A[mid - 1] && A[mid] < A[mid + 1]) {
								start = mid + 1;
							} else if (A[mid] < A[mid - 1] && A[mid] > A[mid + 1]) {
								end = mid - 1;
							} else {
								break;
							}
						}
						int bitonic = mid;
						start = 0, end = bitonic;
						while (start <= end) {
							mid = start + (end - start) / 2;
							if (A[mid] < B) {
								start = mid + 1;
							} else if (A[mid] > B) {
								end = mid - 1;
							} else {
								return mid;
							}
						}
						start = bitonic + 1, end = n - 1;
						while (start <= end) {
							mid = start + (end - start) / 2;
							if (A[mid] > B) {
								start = mid + 1;
							} else if (A[mid] < B) {
								end = mid - 1;
							} else {
								return mid;
							}
						}
						return -1;
					}
				}
				{
					/*
					Smaller or equal elements
					https://www.interviewbit.com/problems/smaller-or-equal-elements/
					*/
					int Solution::solve(vector<int> &A, int B) {
						int l = 0, r = A.size() - 1, count = 0;
						while (l <= r) {
							int mid = l + (r - l) / 2;
							if (A[mid] <= B) {
								count = mid + 1;
								l = mid + 1;
							} else {
								r = mid - 1;
							}
						}
						return count;
					}
				}
				{
					/*
					WoodCutting Made Easy!
					https://www.interviewbit.com/problems/woodcutting-made-easy/

					#Good
					*/
					bool check(vector < int > & A, int val, int B) {
						int rem = 0;
						int n = A.size();
						for (int i = 0; i < n; i++) {
							if (A[i] > val) rem += A[i] - val;
						}
						if (rem >= B) return true;
						return false;
					}
					int Solution::solve(vector < int > & A, int B) {
						int low = 0;
						int high = 1000000;
						int ans = 0;
						while (low <= high) {
							int mid = (low + high) / 2;
							if (check(A, mid, B)) {
								ans = mid;
								low = mid + 1;
							} else high = mid - 1;
						}
						return ans;
					}
				}
				{
					/*
					Matrix Search
					https://www.interviewbit.com/problems/matrix-search/
					*/
					// Better
					{
						int Solution::searchMatrix(vector<vector<int> > &A, int B) {
							int r1 = 0, r2 = A.size() - 1, mid, n = A[0].size();
							while (r1 <= r2) {
								mid = r1 + (r2 - r1) / 2;
								if (A[mid][0] > B) {
									r2 = mid - 1;
								} else if (A[mid][n - 1] < B) {
									r1 = mid + 1;
								} else {
									break;
								}
							}
							vector<int> &D = A[mid];
							int c1 = 0, c2 = n - 1;
							while (c1 <= c2) {
								mid = c1 + (c2 - c1) / 2;
								if (D[mid] == B) {
									return 1;
								} else if (D[mid] > B) {
									c2 = mid - 1;
								} else if (D[mid] < B) {
									c1 = mid + 1;
								}
							}
							return 0;
						}
					}
					// Good
					{
						int Solution::searchMatrix(vector<vector<int> > &A, int B) {
							int N = A.size();
							int M = A[0].size();
							int start = 0, end = N * M - 1;
							while (start <= end) {
								int mid = start + (end - start) / 2;
								int x = mid / M;
								int y = mid % M;
								if (A[x][y] == B) return 1;
								if (B < A[x][y]) end = mid - 1;
								else start = mid + 1;
							}
							return 0;
						}
					}
				}
				{
					/*
					Search for a Range
					https://www.interviewbit.com/problems/search-for-a-range/

					#Revise
					*/
					int search(const vector<int> &A, int B, bool first) {
						int low = 0, high = A.size() - 1, ans = -1;
						while (low <= high) {
							int mid = (low + high) / 2;
							if (A[mid] == B) { ans = mid; if (first) high = mid - 1; else low = mid + 1; }
							else if (A[mid] < B) low = mid + 1;
							else high = mid - 1;
						}
						return ans;
					}
					vector<int> Solution::searchRange(const vector<int> &A, int B) {
						vector<int> result(2);
						result[0] = search(A, B, true);
						result[1] = search(A, B, false);
						return result;
					}
				}
				{
					/*
					Sorted Insert Position
					https://www.interviewbit.com/problems/sorted-insert-position/

					#LowerBound
					*/
					int Solution::searchInsert(vector<int> &A, int B) {
						int l = 0, r = A.size() - 1, ans = A.size();
						while (l <= r) {
							int mid = l + (r - l) / 2;
							if (A[mid] < B) {
								l = mid + 1;
							} else {
								ans = mid;
								r = mid - 1;
							}
						}
						return ans;
					}
				}
			}
			// Search Answer
			{
				{
					/*
					Square Root of Integer
					https://www.interviewbit.com/problems/square-root-of-integer/
					*/
					int Solution::sqrt(int A) {
						if (A == 0 || A == 1) {
							return A;
						}
						int start = 1, end = A / 2;
						while (start <= end) {
							int mid = start + (end - start) / 2;
							if (mid > A / mid) {
								end = mid - 1;
							} else {
								start = mid + 1;
							}
						}
						return end;
					}
				}
				{
					/*
					Allocate Books
					https://www.interviewbit.com/problems/allocate-books/

					#Good #Revise
					*/
					int check(vector<int> &A, int page) {
						int count = 1, n = A.size(), temp = 0;
						for (int i = 0; i < n; i++) {
							temp += A[i];
							if (temp > page) {
								temp = A[i];
								count++;
							}
						}
						return count;
					}
					int Solution::books(vector<int> &A, int B) {
						if (A.size() < B) {
							return -1;
						}
						int start = *max_element(A.begin(), A.end());
						int end = INT_MAX, mid;
						while (start < end) {
							mid = start + (end - start) / 2;
							if (check(A, mid) <= B) {
								end = mid;
							} else {
								start = mid + 1;
							}
						}
						return start;
					}
				}
				{
					/*
					Painter's Partition Problem
					https://www.interviewbit.com/problems/painters-partition-problem/

					#Good #Famous
					*/
					bool check(long long T, int a, vector<int>& c) {
						long long cur = 0;
						int n = c.size(), cnt = 1;
						for (int i = 0; i < n; i++) {
							cur += (long long)(c[i]);
							if (cur > T) {
								cur = (long long)(c[i]);
								cnt++;
							}
						}
						return cnt <= a;
					}
					int Solution::paint(int a, int b, vector<int> &c) {
						long long start = (*max_element(c.begin(), c.end())), end = LONG_MAX, mid;
						while (start <= end) {
							mid = start + (end - start) / 2;
							if (check(mid, a, c)) {
								end = mid - 1;
							} else {
								start = mid + 1;
							}
						}
						long long ans = ((b % 10000003) * (start % 10000003)) % 10000003;
						return ans;
					}
				}
			}
			//Search Step Simulation
			{
				{
					/*
					Implement Power Function
					https://www.interviewbit.com/problems/implement-power-function/

					#Famous
					*/
					int Solution::pow(int x, int n, int d) {
						if (x == 0) return 0;

						int flag = 1;
						if (x < 0) { x = -x; flag = -1;}

						x = x % d;

						if (n == 0) return 1;
						if (n == 1) {return (flag == 1) ? x : (d - x);}

						long long ans;
						long long temp = ((long long int)x * (long long int)x) % d;
						// cout<<n<<" "<<n/2<<" \n";
						if (n % 2 == 0) {
							ans = pow(temp, n / 2 , d) % d;
						} else {
							ans = flag * ((long long int)x % d) * pow(temp, n / 2 , d) % d;
						}
						// cout<<ans%d<<" "<<temp<<" "<<n<<" \n";
						return ans % d;
					}
				}
			}
			// Sort Modification
			{
				{
					/*
					Median of Two Sorted Arrays
					https://leetcode.com/problems/median-of-two-sorted-arrays/

					#Famous
					*/
					double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
						int n1 = nums1.size(), n2 = nums2.size();
						if (n1 < n2) return findMedianSortedArrays(nums2, nums1);
						if (n2 == 0) {
							if (n1 == 1) return nums1[0];
							if (n1 % 2 == 0) {
								return (double)(nums1[n1 / 2 - 1] + nums1[n1 / 2]) / 2;
							}
							return nums1[n1 / 2];
						}
						int total_n = n1 + n2;
						int l1 = 0, r1 = n1 - 1, m1;
						int l2 = 0, r2 = n2 - 1, m2;

						while (l1 <= r1) {
							m1 = (l1 + r1) / 2;
							m2 = (total_n + 1) / 2 - m1 - 2;

							int n1max = (m1 + 1 >= n1) ? INT_MAX : nums1[m1 + 1];
							int n1min = (m1 < 0) ? INT_MIN : nums1[m1];

							int n2max = (m2 + 1 >= n2) ? INT_MAX : nums2[m2 + 1];
							int n2min = (m2 < 0) ? INT_MIN : nums2[m2];

							if (n1min <= n2max && n2min <= n1max) {
								if (total_n % 2 == 0) {
									return (double)(max(n1min, n2min) + min(n1max, n2max)) / 2;
								} else {
									return min(n1max, n2max);
								}
							} else if (n1min > n2max) {
								r1 = m1 - 1;
							} else {
								l1 = m1 + 1;
							}
						}
						return 0;
					}
				}
			}
		}
// Arrays
		{
			// Array Math
			{
				{
					/*
					Pick from both sides!
					https://www.interviewbit.com/problems/pick-from-both-sides/
					*/
					int sum(vector<int> &A, int i, int j) {
						int sum = 0;
						for (int a = i; a <= j; a++) {
							sum += A[a];
						}
						return sum;
					}
					int Solution::solve(vector<int> &A, int B) {
						int max = sum(A, 0, B - 1);
						int temp = max, i = B - 1, j = A.size() - 1;
						for (int k = 1; k <= B; k++) {
							if (i == -1) {
								i = A.size() - 1;
							}
							if (j == -1) {
								j = A.size() - 1;
							}
							temp -= A[i--];
							temp += A[j--];
							if (temp > max) {
								max = temp;
							}
						}
						return max;
					}
				}
				{
					/*
					Min Steps in Infinite Grid
					https://www.interviewbit.com/problems/min-steps-in-infinite-grid/

					#Good
					*/
					int Solution::coverPoints(vector<int> &A, vector<int> &B) {
						int n1 = A.size(), n2 = B.size(), ans = 0;
						for (int i = 1; i < n1; i++) {
							int x = abs(A[i - 1] - A[i]);
							int y = abs(B[i - 1] - B[i]);
							ans += max(x, y);
						}
						return ans;
					}
				}
				{
					/*
					Minimum Lights to Activate
					https://www.interviewbit.com/problems/minimum-lights-to-activate/

					#Good
					*/
					int Solution::solve(vector<int> &A, int B) {
						int b = B - 1, count = 0;
						int s = A.size();
						int lastlight = -1, check = 0;
						for (int i = b; i >= 0; i--) {
							if (A[i] == 1) {
								lastlight = i; //last light from B-1 to 0
								count++;
								break;
							}
						}
						if (count == 0) {
							return -1;
						}
						check = lastlight + B; //first lightoff
						while (check < s) {
							int flag = 0;
							for (int i = min(check + b, s - 1); i >= check - b; i--) {
								if (A[i] == 1) {
									lastlight = i;
									flag = 1;
									check = i + B;
									count++;
									break;
								}
							}
							if (flag == 0) {
								return -1;
							}
						}
						return count;
					}
				}
				{
					/*
					Max Sum Contiguous Subarray
					https://www.interviewbit.com/problems/max-sum-contiguous-subarray/

					#Famous #Revise
					*/
					int Solution::maxSubArray(const vector<int> &A) {
						int mx = *max_element(A.begin(), A.end());
						if (mx <= 0)return mx;
						int ans = 0, sum = 0;
						for (int i = 0; i < A.size(); i++) {
							sum += A[i];
							if (sum <= 0) {
								sum = 0;
								continue;
							}
							ans = max(ans, sum);
						}
						return ans;
					}
				}
				{
					/*
					Maximum Absolute Difference
					https://www.interviewbit.com/problems/maximum-absolute-difference/

					#Good #Tricky
					*/
					int Solution::maxArr(vector<int> &A) {
						int n = A.size();
						vector<int> P(n), N(n);
						int ans = 0;
						for (int i = 0; i < n; i++) {
							P[i] = A[i] + i;
							N[i] = A[i] - i;
						}
						return max(*max_element(P.begin(), P.end()) - *min_element(P.begin(), P.end()),
						           *max_element(N.begin(), N.end()) - *min_element(N.begin(), N.end()));
					}
				}
				{
					/*
					Partitions
					https://www.interviewbit.com/problems/partitions/

					#Good
					*/
					// Best
					{
						// Function to count the no of ways
						int countways(int n, vector<int>& a)
						{
							int pre[n];
							pre[0] = a[0];
							for (int i = 1; i < n; i++) pre[i] = pre[i - 1] + a[i];
							if (pre[n - 1] % 3) return 0;
							int req = pre[n - 1] / 3;
							int cnt = 0;
							int ans = 0;
							for (int i = 0; i < n - 1; i++)
							{
								if (pre[i] == 2 * req) ans += cnt;
								cnt += (pre[i] == req);
							}
							return ans;
						}
						int Solution::solve(int A, vector<int> &B) {
							assert(A >= 1 && A <= 100000 && A == B.size());
							for (int a : B)assert(a >= -1000000000 && a <= 1000000000);
							return countways(A, B);
						}
					}
					// Good
					{
						int Solution::solve(int n, vector<int> &A) {
							vector<int> sum(n, 0);
							sum[0] = A[0];
							for (int i = 1; i < n; i++) {
								sum[i] = sum[i - 1] + A[i];
							}
							if (abs(sum[n - 1]) % 3 > 0)return 0;
							int count = 0, S = sum[n - 1] / 3;
							for (int i = 0; i < n - 1; i++) {
								for (int j = i + 1; j < n - 1; j++) {
									if (sum[i] == S && sum[j] == 2 * S)count++;
								}
							}
							return count;
						}
					}
				}
				{
					/*
					Maximum Area of Triangle!
					https://www.interviewbit.com/problems/maximum-area-of-triangle/
					*/
					int ans = 0;
					int code(char c) {
						if (c == 'r') {
							return 0;
						} else if (c == 'g') {
							return 1;
						} else {
							return 2;
						}
					}
					int area(float x, float y) {
						return ceil((x * y) / 2);
					}
					int Solution::solve(vector<string> &A) {
						ans = 0;
						int m = A[0].size(), n = A.size();
						vector<int> M(3, -1), // largest overall
						       N(3, INT_MAX); //smallest overall
						for (int i = 0; i < n; i++) {
							for (int j = 0; j < m; j++) {
								int c = code(A[i][j]);
								M[c] = max(M[c], j);
								N[c] = min(N[c], j);
							}
						}
						for (int j = 0; j < m; j++) {
							vector<int> B(3, -1), //largest in col
							       S(3, INT_MAX); //smallest in col
							for (int i = 0; i < n; i++) {
								int c = code(A[i][j]);
								B[c] = max(B[c], i);
								S[c] = min(S[c], i);
							}
							for (int c = 0; c < 3; c++) {
								int a = (c + 1) % 3, b = (c + 2) % 3;
								if (B[a] != -1 && B[b] != -1 && M[c] != -1) {
									int x = max(abs(B[a] - S[b]), abs(B[b] - S[a])) + 1;
									int y = max(abs(M[c] - j), abs(N[c] - j)) + 1;
									if (y == 1)continue;
									ans = max(ans, area(x, y));
								}
							}
						}
						return ans;
					}
				}
				{
					/*
					Flip
					https://www.interviewbit.com/problems/flip/

					#Good #VeryGood
					*/
					vector<int> Solution::flip(string A) {
						int n = A.length(), start = 1, end = -1;
						int sum = 0, psum = 0;
						vector<int> vec{1, -1};
						for (int i = 0; i < A.size(); i++) {
							if (A[i] == '0') {
								sum++;
								if (sum > psum) {
									vec[0] = start;
									vec[1] = i + 1;
									psum = sum;
								}
							} else if (A[i] == '1') {
								sum--;
								if (sum < 0) {
									sum = 0;
									start = i + 2;
								}
							}
						}

						if (vec[1] == -1) {
							vec.clear();
						}
						return vec;
					}
				}
			}
			// Value Range
			{
				{
					/*
					Merge Intervals
					https://www.interviewbit.com/problems/merge-intervals/
					*/
					vector<Interval> Solution::insert(vector<Interval> &intervals, Interval newInterval) {
						vector<Interval> res;
						for (Interval test : intervals) {
							if (newInterval.start > test.end) res.push_back(test);
							else if (test.start > newInterval.end) {
								res.push_back(newInterval);
								newInterval = test;
							} else if (newInterval.start <= test.end || newInterval.end >= test.start) {
								newInterval = Interval(min(test.start, newInterval.start), max(test.end, newInterval.end));
							}
						}
						res.push_back(newInterval);
						return res;
					}
				}
				{
					/*
					Merge Overlapping Intervals
					https://www.interviewbit.com/problems/merge-overlapping-intervals/

					#Good
					*/
					bool sortfun(Interval A, Interval B) {
						return (A.start < B.start);
					}
					vector<Interval> Solution::merge(vector<Interval> &A) {
						int n = A.size();
						sort(A.begin(), A.end(), sortfun);
						vector<Interval> ans;
						Interval temp;
						temp = Interval(A[0].start, A[0].end);
						for (int i = 1; i < n; i++) {
							if (A[i].start >= temp.start && A[i].start <= temp.end) {
								temp = Interval(min(A[i].start, temp.start), max(A[i].end, temp.end));
							} else {
								ans.push_back(temp);
								temp = Interval(A[i].start, A[i].end);
							}
						}
						ans.push_back(temp);

						return ans;
					}
				}
				{
					/*
					Perfect Peak of Array
					https://www.interviewbit.com/problems/perfect-peak-of-array/
					*/
					int Solution::perfectPeak(vector<int> &A) {
						int size = A.size(), leftmax[size], rightmin[size];
						leftmax[0] = A[0];
						for (int i = 1; i < size; i++) {
							leftmax[i] = max(A[i - 1], leftmax[i - 1]);
						}
						rightmin[size - 1] = A[size - 1];
						for (int i = size - 2; i >= 0; i--) {
							rightmin[i] = min(A[i + 1], rightmin[i + 1]);
						}
						for (int i = 1; i < size - 1; i++) {
							if (A[i] > leftmax[i] && A[i] < rightmin[i]) {
								return 1;
							}
						}
						return 0;
					}
				}
				{
					/*
					Spiral Order Matrix II
					https://www.interviewbit.com/problems/spiral-order-matrix-ii/
					*/
					//Best Solution
					{
						vector<vector<int> > Solution::generateMatrix(int A) {
							vector< vector<int> > arr(A, vector<int> (A, 0));
							int start, end, i;
							long long int x;
							x = 1;
							for (start = 0, end = A - 1; start <= end; start++, end--)
							{
								for (i = start; i <= end; i++) { arr[start][i] = x; x++;}
								for (i = start + 1; i <= end; i++) { arr[i][end] = x; x++;}
								for (i = end - 1; i >= start; i--) { arr[end][i] = x; x++;}
								for (i = end - 1; i >= start + 1; i--) { arr[i][start] = x; x++;}
							}
							return arr;
						}
					}
					// Good
					{
						vector<vector<int> > Solution::generateMatrix(int A) {
							vector<vector<int>> B(A, vector<int>(A, 0));
							int num = 1;
							int lim[4] = {0, A - 1, A - 1, 0}; //top 0, right 1, bottom 2, left 3
							int direction = 1;

							while (lim[0] <= lim[2] && lim[3] <= lim[1]) {
								if (direction == 1) {
									for (int i = lim[3]; i <= lim[1]; i++) {
										B[lim[0]][i] = num++;
									}
									lim[0]++;
									direction = 2;
								} else if (direction == 2) {
									for (int i = lim[0]; i <= lim[2]; i++) {
										B[i][lim[1]] = num++;
									}
									lim[1]--;
									direction = 3;
								} else if (direction == 3) {
									for (int i = lim[1]; i >= lim[3]; i--) {
										B[lim[2]][i] = num++;
									}
									lim[2]--;
									direction = 4;
								} else if (direction == 4) {
									for (int i = lim[2]; i >= lim[0]; i--) {
										B[i][lim[3]] = num++;
									}
									lim[3]++;
									direction = 1;
								}
							}
							return B;
						}
					}
				}
				{
					/*
					Pascal Triangle
					https://www.interviewbit.com/problems/pascal-triangle/
					*/
					vector<vector<int> > Solution::solve(int A) {
						vector<vector<int>> B;
						if (A == 0) {
							return B;
						}
						vector<int> C(1, 1);
						B.push_back(C);
						for (int i = 1; i < A; i++) {
							vector<int> C(i + 1, 1);
							for (int j = 1; j < i; j++) {
								C[j] = B[i - 1][j] + B[i - 1][j - 1];
							}
							B.push_back(C);
						}
						return B;
					}
				}
				{
					/*
					Anti Diagonals
					https://www.interviewbit.com/problems/anti-diagonals/
					*/
					vector<vector<int> > Solution::diagonal(vector<vector<int> > &A) {
						int n = A.size();
						if (n == 0) {
							return {} ;
						}
						vector<vector<int>> B(2 * n - 1);
						for (int k = 0; k < n; k++) {
							vector<int> C(k + 1);
							int i = 0, j = k;
							while (i <= k) {
								C[i] = A[i][j--];
								i++;
							}
							B[k] = C;
						}
						for (int k = 0; k < n - 1; k++) {
							vector<int> C(n - 1 - k);
							int i = 1 + k, j = n - 1;
							while (i < n) {
								C[i - 1 - k] = A[i][j--];
								i++;
							}
							B[k + n] = C;
						}
						return B;
					}
				}
				{
					/*
					Triplets with Sum between given range
					https://www.interviewbit.com/problems/triplets-with-sum-between-given-range/

					#Good
					*/
					int Solution::solve(vector<string> &A) {
						int n = A.size();
						vector<double> v(n, 0);
						for (int i = 0; i < n; i++) v[i] = stod(A[i]);

						if (n < 3) return 0;

						double minm, maxm;
						bool lower_limit, uper_limit;
						maxm = (v[0] > v[1]) ? v[0] : v[1];
						minm = (v[0] < v[1]) ? v[0] : v[1];
						double currsum;
						for (int i = 2; i < n; i++) {
							currsum = maxm + minm + v[i];
							lower_limit = currsum > 1;
							uper_limit = currsum < 2;

							if (lower_limit && uper_limit) return 1;
							else if (!uper_limit) {
								maxm = min(maxm, v[i]);
								if (minm > maxm) swap<double>(minm, maxm);
							}
							else if (!lower_limit) {
								minm = max(minm, v[i]);
								if (minm > maxm) swap<double>(minm, maxm);
							}

						}
						return 0;
					}
				}
				{
					/*
					Balance Array
					https://www.interviewbit.com/problems/balance-array/
					*/
					int Solution::solve(vector<int> &A) {
						int size = A.size(), count = 0;
						int lefteven = 0, leftodd = 0;
						int righteven = 0, rightodd = 0;
						for (int i = 0; i < size; i += 2) {
							righteven += A[i];
						}
						for (int i = 1; i < size; i += 2) {
							rightodd += A[i];
						}

						for (int i = 0; i < size; i++) {
							//cout<<lefteven<<" "<<rightodd<<" "<<leftodd<<" "<<righteven<<" ";
							if (i % 2 == 0) {
								if ((lefteven + rightodd) == (leftodd + righteven - A[i])) {
									count++;
								}
								lefteven += A[i];
								righteven -= A[i];
							}
							if (i % 2 == 1) {
								if ((lefteven + rightodd - A[i]) == (leftodd + righteven)) {
									count++;
								}
								leftodd += A[i];
								rightodd -= A[i];
							}
						}
						return count;
					}
				}
				{
					/*
					Find Duplicate in Array
					https://www.interviewbit.com/problems/find-duplicate-in-array/
					*/
					int Solution::repeatedNumber(const vector<int> &A) {
						vector<int> B(A.size(), 0);
						for (int i = 0; i < A.size(); i++) {
							B[A[i]] += 1;
						}
						for (int i = 0; i < A.size(); i++) {
							if (B[i] >= 2) {
								return i;
							}
						}
						return -1;
					}
				}
				{
					/*
					Maximum Consecutive Gap
					https://www.interviewbit.com/problems/maximum-consecutive-gap/

					#Good #Revise #Bucketing
					*/
					int Solution::maximumGap(const vector<int> &A) {
						if (A.size() < 2) return 0;
						int mn = *min_element(A.begin(), A.end()), mx = *max_element(A.begin(), A.end()), n = A.size();
						double gap = (double)(mx - mn) / (n - 1);
						if (gap == 0)return 0;
						vector<int>mx_buc(n, INT_MIN), mn_buc(n, INT_MAX);
						for (int x : A) {
							int ind = floor((x - mn) / gap);
							if (mn == x)continue;
							mx_buc[ind] = max(mx_buc[ind], x);
							mn_buc[ind] = min(mn_buc[ind], x);
						}
						int mx_gp = INT_MIN, prev_mx = mn;
						for (int i = 0; i < n; i++) {
							if (mn_buc[i] == INT_MAX)continue;
							int cur = mn_buc[i] - prev_mx;
							mx_gp = max(mx_gp, cur);
							prev_mx = mx_buc[i];
						}
						return mx_gp;
					}
				}
			}
			// Arrangement
			{
				{
					/*
					Sort array with squares!
					https://www.interviewbit.com/problems/sort-array-with-squares/
					*/
					vector<int> Solution::solve(vector<int> &A) {
						int s = A.size(), start = 0, end = s - 1;
						int sq1, sq2;
						vector<int> B;
						sq1 = A[start] * A[start];
						sq2 = A[end] * A[end];
						while (start != end) {
							if (sq1 >= sq2) {
								B.push_back(sq1);
								start++;
								sq1 = A[start] * A[start];
							} else {
								B.push_back(sq2);
								end--;
								sq2 = A[end] * A[end];
							}
						}
						B.push_back(A[start]*A[start]);
						reverse(B.begin(), B.end());
						return B;
					}
				}
				{
					/*
					Largest Number
					https://www.interviewbit.com/problems/largest-number/

					#Good
					*/
					int myCompare(string X, string Y) {
						string XY = X.append(Y);
						string YX = Y.append(X);
						return XY.compare(YX) > 0 ? 1 : 0;
					}

					string Solution::largestNumber(const vector<int> &A) {
						vector<string> b;
						for (int i = 0; i < A.size(); i++) {
							b.push_back(to_string(A[i]));
						}
						sort(b.begin(), b.end(), myCompare);
						string ans = "";
						for (int i = 0; i < b.size(); i++) {
							ans += b[i];
						}
						int i = 0;
						while (ans[i] == '0') {
							i++;
						}
						if (i == ans.length())
							ans = "0";
						return ans;
					}
				}
				{
					/*
					Rotate Matrix
					https://www.interviewbit.com/problems/rotate-matrix/
					*/
					void Solution::rotate(vector<vector<int> > &A) {
						int n = A.size();
						for (int i = 0; i < n; i++) {
							for (int j = i; j < n - i - 1; j++) {
								swap(A[i][j], A[j][n - i - 1]);
								swap(A[i][j], A[n - i - 1][n - j - 1]);
								swap(A[i][j], A[n - j - 1][i]);
							}
						}
					}
				}
				{
					/*
					Find Permutation
					https://www.interviewbit.com/problems/find-permutation/
					*/
					vector<int> Solution::findPerm(const string A, int B) {
						int big = B, small = 1, i = 0;
						vector<int> ans(B);
						while (i < B) {
							if (A[i] == 'D') {
								ans[i++] = big--;
							} else {
								ans[i++] = small++;
							}
						}
						return ans;
					}
				}
				{
					/*
					Next Permutation
					https://www.interviewbit.com/problems/next-permutation/

					#Good
					*/
					vector<int> Solution::nextPermutation(vector<int> &num) {
						int len = num.size();
						int i, j;
						for (i = len - 2; i >= 0 ; i--)
							if (num[i] < num[i + 1]) break;

						if (i == -1) {
							reverse(num.begin(), num.end());
							return num;
						}

						for (j = len - 1; j > i; j--)
							if (num[j] > num[i]) break;

						swap(num[i], num[j]);
						reverse(num.begin() + i + 1, num.end());
						return num;
					}
				}
			}
			// Sorting
			{
				{
					/*
					Noble Integer
					https://www.interviewbit.com/problems/noble-integer/
					*/
					int Solution::solve(vector<int> &A) {
						int n = A.size();
						sort(A.begin(), A.end(), greater<int>());
						if (A[0] == 0) return 1;
						for (int i = 1; i < n; i++) {
							if (A[i] == i && A[i] != A[i - 1]) return 1;
						}
						return -1;
					}
				}
				{
					/*
					Wave Array
					https://www.interviewbit.com/problems/wave-array/
					*/
					vector<int> Solution::wave(vector<int> &A) {
						sort(A.begin(), A.end());
						int i = 0;
						for (i = 0; i < A.size() - 2; i += 2) {
							swap(A[i], A[i + 1]);
						}
						if (i == A.size() - 2) {
							swap(A[A.size() - 1], A[A.size() - 2]);
						}
						return A;
					}
				}
				{
					/*
					Hotel Bookings Possible
					https://www.interviewbit.com/problems/hotel-bookings-possible/

					#Good
					*/
					bool Solution::hotel(vector<int> &arrival, vector<int> &departure, int K) {
						int n = arrival.size();
						vector<pair<int, int> > ans;
						for (int i = 0; i < n; i++) {
							ans.push_back(make_pair(arrival[i], 1));
							ans.push_back(make_pair(departure[i], 0));
						}

						sort(ans.begin(), ans.end());

						int curr_active = 0, max_active = 0;

						for (int i = 0; i < ans.size(); i++) {
							if (ans[i].second == 1) {
								curr_active++;
								max_active = max(max_active, curr_active);
							}
							else
								curr_active--;
						}

						return (K >= max_active);
					}
				}
				{
					/*
					Max Distance
					https://www.interviewbit.com/problems/max-distance/
					*/
					int Solution::maximumGap(const vector<int> &A) {
						int s = A.size();
						vector<pair<int, int>> B;
						for (int i = 0; i < s; i++) {
							B.push_back(make_pair(A[i], i));
						}
						sort(B.begin(), B.end());
						int maxIndex = B[s - 1].second;
						int ans = 0;
						for (int i = s - 2; i >= 0; i--) {
							ans = max(ans, maxIndex - B[i].second);
							maxIndex = max(maxIndex, B[i].second);
						}
						return ans;
					}
				}
				{
					/*
					Maximum Unsorted Subarray
					https://www.interviewbit.com/problems/maximum-unsorted-subarray/
					*/
					vector<int> Solution::subUnsort(vector<int> &A) {
						int n = A.size();
						vector<int> ans(1, -1);
						if (n <= 1) {
							return ans;
						}
						int max = A[0];
						int i;
						for (i = 1; i < n; i++) {
							if (A[i] < A[i - 1]) {
								ans[0] = i - 1;
								break;
							}
						}
						if (i == n)   return ans;
						ans.push_back(n - 1);
						for (i = n - 2; i >= ans[0]; i--) {
							if (A[i] > A[i + 1]) {
								ans[1] = i + 1;
								break;
							}
						}
						int mini = *min_element(A.begin() + ans[0], A.begin() + ans[1] + 1);
						int maxi = *max_element(A.begin() + ans[0], A.begin() + ans[1] + 1);
						for (i = 0; i < ans[0]; i++) {
							if (A[i] > mini) {
								ans[0] = i;
							}
						}
						for (i = n - 1; i > ans[1]; i--) {
							if (A[i] < maxi) {
								ans[1] = i;
							}
						}
						return ans;
					}
				}
			}
			// Space Recycle
			{
				{
					/*
					Set Matrix Zeros
					https://www.interviewbit.com/problems/set-matrix-zeros/

					#Famous
					*/
					void Solution::setZeroes(vector<vector<int> > &A) {
						int n = A.size();
						int m = A[0].size();

						vector<bool> row(n, false), col(m, false);
						for (int i = 0; i < n; i++) {
							for (int j = 0; j < m; j++) {
								if (A[i][j] == 0) {
									row[i] = true;
									col[j] = true;
								}
							}
						}

						for (int i = 0; i < n; i++) {
							for (int j = 0; j < m; j++) {
								if (row[i] || col[j]) {
									A[i][j] = 0;
								}
							}
						}
					}
				}
				{
					/*
					Maximum Sum Square SubMatrix
					https://www.interviewbit.com/problems/maximum-sum-square-submatrix/
					*/
				}
			}
			// Missing / Repeated number
			{
				{
					/*
					First Missing Integer
					https://www.interviewbit.com/problems/first-missing-integer/

					#Good #Famous #Revise
					*/
					int Solution::firstMissingPositive(vector<int> &A) {
						int n = A.size();
						for (int i = 0; i < n; i++) {
							if (A[i] > 0 && A[i] <= n) {
								int pos = A[i] - 1;
								if (A[pos] != A[i]) {
									swap(A[pos], A[i]);
									i--;
								}
							}
						}
						for (int i = 0; i < n; i++) {
							if (A[i] != i + 1) return (i + 1);
						}
						return n + 1;
					}
				}
				{
					/*
					Repeat and Missing Number Array
					https://www.interviewbit.com/problems/repeat-and-missing-number-array/

					#Tricky #Revise
					*/
					//#define ll long long
					vector<int> Solution::repeatedNumber(const vector<int> &A) {
						ll n = A.size();

						ll diff = 0;
						for (ll i = 0; i < n; i++)  diff += (ll)(A[i] - (i + 1)); // a-b

						ll sqDiff = 0;
						for (ll i = 0; i < n; i++)  sqDiff += (ll)((ll)A[i] * (ll)A[i] - (i + 1) * (i + 1)); //I was forgetting to typecast here, so It gave alot of error :(

						ll sum = sqDiff / diff; // a+b

						int a = (sum + diff) / 2;
						int b = (sum - a);

						vector<int> v{a, b};
						return v;
					}
				}
				{
					/*
					N/3 Repeat Number
					https://www.interviewbit.com/problems/n3-repeat-number/

					#Tricky #Famous
					*/
					int count(int x, const vector<int> &A) {
						int n = A.size(), cont = 0;
						for (int i = 0; i < n; i++) {
							if (A[i] == x)cont++;
						}
						return cont;
					}
					int Solution::repeatedNumber(const vector<int> &A) {
						int n = A.size(), lim = n / 3 + 1;
						vector<pair<int, int>> M;
						for (int i = 0; i < n; i++) {
							if (M.empty()) {
								M.push_back({A[i], 1});
							} else if (A[i] == M[0].first) {
								M[0].second++;
							} else if (M.size() == 1) {
								M.push_back({A[i], 1});
							} else if (A[i] == M[1].first) {
								M[1].second++;
							} else {
								M[0].second--;
								M[1].second--;
								if (M[0].second == 0 && M[1].second == 0) {
									M.clear();
								} else if (M[0].second == 0) {
									M.erase(M.begin());
								} else if (M[1].second == 0) {
									M.erase(M.begin() + 1);
								}
							}
						}
						for (auto x : M) {
							if (count(x.first, A) >= lim)return x.first;
						}
						return -1;
					}
				}
			}
		}
// Strings
		{
			// String Simulation
			{
				{
					/*
					Palindrome String
					https://www.interviewbit.com/problems/palindrome-string/

					#Famous
					*/
					int Solution::isPalindrome(string s) {
						int i = 0, j = (int)s.size() - 1;
						while (i < j) {
							while (i < j && !isalnum(s[i])) i++;
							while (i < j && !isalnum(s[j])) j--;
							if (toupper(s[i]) != toupper(s[j])) return false;
							i++;
							j--;
						}
						return true;
					}
				}
				{
					/*
					Vowel and Consonant Substrings!
					https://www.interviewbit.com/problems/vowel-and-consonant-substrings/
					*/
					bool vowel(char c) {
						if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')return 1;
						return 0;
					}
					int Solution::solve(string A) {
						int n = A.length(), vcount = 0, ccount = 0, ans = 0;
						for (int i = 0; i < n; i++) {
							if (vowel(A[i])) {
								vcount++;
								ans += ccount;
							} else {
								ccount++;
								ans += vcount;
							}
							ans %= 1000000007;
						}
						return ans;
					}
				}
				{
					/*
					Remove Consecutive Characters
					https://www.interviewbit.com/problems/remove-consecutive-characters/
					*/
					string Solution::solve(string A, int B) {
						char temp = A[0];
						int count = 0;
						string ans = "";
						for (int i = 0; i < A.length(); i++) {
							if (temp != A[i]) {
								if (count != B) ans += string(count, temp);
								count = 1;
								temp = A[i];
							} else {
								count++;
							}
						}
						if (count != B) ans += string(count, temp);
						return ans;
					}
				}
				{
					/*
					Longest Common Prefix
					https://www.interviewbit.com/problems/longest-common-prefix/

					#Good
					*/
					// Best
					{
						Can use Binary search on minimum size string
					}
					// Good
					{
						string Solution::longestCommonPrefix(vector<string> &A) {
							string ans = "";
							int n = A.size(), m = INT_MAX;
							for (string s : A) m = min(m, (int) s.size());
							for (int j = 0; j < m; j++) {
								for (int i = 1; i < n; i++) {
									if (A[i][j] != A[0][j]) return ans;
								}
								ans += A[0][j];
							}
							return ans;
						}
					}
				}
				{
					/*
					Count And Say
					https://www.interviewbit.com/problems/count-and-say/
					*/
					string Solution::countAndSay(int A) {
						string ans = "1";
						while (A-- > 1) {
							int n = ans.size(), count = 1;
							string temp = "";
							for (int i = 1; i < n; i++) {
								if (ans[i] != ans[i - 1]) {
									temp += to_string(count) + ans[i - 1];
									count = 1;
								} else {
									count++;
								}
							}
							temp += to_string(count) + ans.back();
							ans = temp;
						}
						return ans;
					}
				}
			}
			// String Search
			{
				{
					/*
					Amazing Subarrays
					https://www.interviewbit.com/problems/amazing-subarrays/
					*/
					bool check(char c) {
						c = toupper(c);
						if (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U')return 1;
						return 0;
					}
					int Solution::solve(string A) {
						int n = A.length(), count = 0;
						for (int i = 0; i < n; i++) {
							if (check(A[i])) {
								count += (n - i) % 10003;
							}
						}
						return count % 10003;
					}
				}
				{
					/*
					Implement StrStr
					https://www.interviewbit.com/problems/implement-strstr/

					#KMP
					*/
				}
				{
					/*
					Stringoholics
					https://www.interviewbit.com/problems/stringoholics/

					#GCD*/
				}
			}
			// String Tricks
			{
				{
					/*
					Minimum Characters required to make a String Palindromic
					https://www.interviewbit.com/problems/minimum-characters-required-to-make-a-string-palindromic/
					*/
					int Solution::solve(string A) {
						int n = A.length();
						int l = 0, r = n - 1;
						while (l < r) {
							if (A[l] != A[r]) {
								if (l == 0) {
									r--;
								} else {
									l = 0;
								}
							} else {
								l++;
								r--;
							}
						}
						if (l == r) {
							return n - 2 * l - 1;
						} else {
							return n - 2 * l;
						}
					}
				}
				{
					/*
					Longest Palindromic Substring
					https://www.interviewbit.com/problems/longest-palindromic-substring/

					#Famous #Revise
					*/
					string expandAroundCenter(string s, int c1, int c2) {
						int l = c1, r = c2;
						int n = s.length();
						while (l >= 0 && r <= n - 1 && s[l] == s[r]) {
							l--;
							r++;
						}
						return s.substr(l + 1, r - l - 1);
					}
					string Solution::longestPalindrome(string s) {
						int n = s.length();
						if (n == 0) return "";
						string longest = s.substr(0, 1);  // a single char itself is a palindrome
						for (int i = 0; i < n - 1; i++) {
							string p1 = expandAroundCenter(s, i, i);
							if (p1.length() > longest.length())
								longest = p1;

							string p2 = expandAroundCenter(s, i, i + 1);
							if (p2.length() > longest.length())
								longest = p2;
						}
						return longest;
					}
				}
				{
					/*
					Minimum Parantheses!
					https://www.interviewbit.com/problems/minimum-parantheses/
					*/
					int Solution::solve(string A) {
						int start = 0, i = 0, count = 0;
						while (A[i] != '\0') {
							if (A[i] == '(') {
								start++;
							} else if (A[i] == ')') {
								if (start > 0) {
									start--;
								} else {
									count++;
								}
							}
							i++;
						}
						return start + count;
					}
				}
				{
					/*
					Minimum Appends for Palindrome!
					https://www.interviewbit.com/problems/minimum-appends-for-palindrome/

					#Good
					*/
					int Solution::solve(string A) {
						string B = A;
						reverse(B.begin(), B.end());
						B = B + '#' + A;
						int n = B.length(), i = 1, len = 0;
						vector<int> LPS(n, 0);
						while (i < n) {
							if (B[len] == B[i]) {
								len++;
								LPS[i] = len > (i + 1) / 2 ? (i + 1) / 2 : len;
							} else {
								if (len == 0) {
									LPS[i] = 0;
								} else {
									len = LPS[len - 1];
									i--;
								}
							}
							i++;
						}
						return A.length() - LPS.back();
					}
				}
				{
					/*
					Convert to Palindrome
					https://www.interviewbit.com/problems/convert-to-palindrome/
					*/
					int Solution::solve(string A) {
						int left = 0;
						int right = A.size() - 1;
						int count = 0;
						while (left < right)
						{
							if (A[left] == A[right]) {left++; right--;}
							else if (A[left + 1] == A[right]) {count++; left++;}
							else if (A[left] == A[right - 1]) {count++; right--;}
							else return 0;
						}
						if (count == 0 and left == right) return 1;
						if (count == 1) return 1;
						return 0;
					}
				}
			}
			// String Math
			{
				{
					/*
					Integer To Roman
					https://www.interviewbit.com/problems/integer-to-roman/

					#Famous
					*/
					vector<char> R = {'I', 'V', 'X', 'L', 'C', 'D', 'M'};
					string roman(int x, int p) {
						if (x == 0)return "";
						if (x <= 3)return string(x, R[2 * p]);
						else if (x <= 5)return string(5 - x, R[2 * p]) + R[2 * p + 1];
						else if (x <= 8)return R[2 * p + 1] + string(x - 5, R[2 * p]);
						string S = "";
						return S + R[2 * p] + R[2 * p + 2];
					}
					string Solution::intToRoman(int A) {
						int num[4];
						string S = "";
						for (int i = 3; i >= 0; i--) {
							S += roman((A % (int)pow(10, i + 1)) / pow(10, i), i);
						}
						return S;
					}
				}
				{
					/*
					Roman To Integer
					https://www.interviewbit.com/problems/roman-to-integer/

					#Famous
					*/
					int romanCharToInt(char c) {
						switch (c) {
						case 'I':   return 1;
						case 'V':   return 5;
						case 'X':   return 10;
						case 'L':   return 50;
						case 'C':   return 100;
						case 'D':   return 500;
						case 'M':   return 1000;
						default:    return 0;
						}
					}
					int Solution::romanToInt(string s) {
						int num = 0;
						int size = s.size();

						for (int i = 0; i < size; i++) {
							if (i < (size - 1) && romanCharToInt(s[i]) < romanCharToInt(s[i + 1])) {
								num -= romanCharToInt(s[i]);
							} else {
								num += romanCharToInt(s[i]);
							}
						}
						return num;
					}
				}
				{
					/*
					Add Binary Strings
					https://www.interviewbit.com/problems/add-binary-strings/
					*/
					string Solution::addBinary(string A, string B) {
						if (A.length() < B.length()) {
							swap(A, B);
						}
						int n = A.length(), m = B.length();
						B = string(n - m, '0') + B;
						int carry = 0, temp = 0;
						for (int i = n - 1; i >= 0; i--) {
							temp = A[i] - '0' + B[i] - '0' + carry;
							if (temp == 3) {
								carry = 1;
							} else if (temp == 2) {
								A[i] = '0';
								carry = 1;
							} else {
								A[i] = '0' + temp;
								carry = 0;
							}
						}
						if (carry == 1) {
							A = '1' + A;
						}
						return A;
					}
				}
				{
					/*
					Power of 2
					https://www.interviewbit.com/problems/power-of-2/
					*/
					string half(string A) {
						string B = A;
						int carry = 0;
						for (int i = 0; i < A.length(); i++) {
							B[i] = '0' + (A[i] - '0' + carry) / 2;
							carry = ((A[i] - '0' + carry) % 2) * 10;
						}
						if (B[0] == '0') {
							return B.substr(1, B.length() - 1);
						}
						return B;
					}
					int Solution::power(string A) {
						int n = A.size();
						int temp = A[A.length() - 1] - '0';
						while (temp % 2 == 0) {
							A = half(A);
							if (A == "1") {
								return 1;
							}
							temp = A[A.length() - 1] - '0';
						}
						return 0;
					}
				}
				{
					/*
					Multiply Strings
					https://www.interviewbit.com/problems/multiply-strings/
					*/
					string Solution::multiply(string A, string B) {
						int m = A.size(), n = B.size();
						string ans(m + n, '0');

						for (int i = m - 1; i >= 0; --i) {
							for (int j = n - 1; j >= 0; --j) {
								int sum = (A[i] - '0') * (B[j] - '0') + (ans[i + j + 1] - '0');
								ans[i + j + 1] = (sum % 10) + '0';
								ans[i + j] += (sum / 10);
							}
						}

						for (int i = 0; i < m + n; ++i) {
							if (ans[i] != '0')
								return ans.substr(i);
						}
						return "0";
					}
				}
			}
			// String Parsing
			{
				{
					/*
					Compare Version Numbers
					https://www.interviewbit.com/problems/compare-version-numbers/

					#Good
					*/
					int Solution::compareVersion(string A, string B) {
						int j, i;
						for ( i = 0, j = 0 ; i < A.size() || j < B.size() ; i++, j++) {
							unsigned long long num1 = 0, num2 = 0;
							while (i < A.size() && A[i] != '.') {
								num1 *= 10;
								num1 += A[i] - '0';
								i++;
							}
							while (j < B.size() && B[j] != '.') {
								num2 *= 10;
								num2 += B[j] - '0';
								j++;
							}
							if (num1 > num2) return 1;
							if (num1 < num2) return -1;
						}
						return 0;
					}
				}
				{
					/*
					Atoi
					https://www.interviewbit.com/problems/atoi/

					#Revise
					*/
					int Solution::atoi(const string & str) {
						if (str == "") return 0;
						stringstream ss(str);
						int ret;
						ss >> ret;
						return ret;
					}
				}
			}
		}
// Math
		{
			// Adhoc
			{
				{
					/*
					Verify Prime
					https://www.interviewbit.com/problems/verify-prime/
					*/
					int Solution::isPrime(int A) {
						if (A <= 1) {
							return 0;
						}
						int s = (int) sqrt(A);
						for (int i = 2; i <= s; i++) {
							if (A % i == 0) {
								return 0;
							}
						}
						return 1;
					}
				}
				{
					/*
					Sum of pairwise Hamming Distance
					https://www.interviewbit.com/problems/sum-of-pairwise-hamming-distance/

					#Good
					*/
					long long mod = 1000000007;
					int Solution::hammingDistance(const vector<int> &A) {
						long long ans = 0, x = 1, n = A.size();
						while (x > 0) {
							long long count = 0;
							for (int i = 0; i < n; i++) {
								if ((A[i]&x) > 0)count++;
							}
							ans += 2LL * (count * (n - count));
							ans %= mod;
							x <<= 1;
						}
						return ans;
					}
				}
				{

				}
			}
		}
	}







































































}