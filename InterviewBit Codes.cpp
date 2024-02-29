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
					vis[ind] = 0;
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

				#Famous #Bipartate #2Color
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
				// White Grey Balck Algo : https://leetcode.com/problems/course-schedule/?envType=study-plan-v2&envId=top-interview-150
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

				#Good #VeryGood
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

}

{
	/*
	*/
}
