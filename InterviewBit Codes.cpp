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
					if (dp[s][e] != -1)return dp[s][e];
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
				*/
			}
			{
				/*
				*/
			}
			{
				/*
				*/
			}
			{
				/*
				*/
			}
			{
				/*
				*/
			}
			{
				/*
				*/
			}
			{
				/*
				*/
			}
			{
				/*
				*/
			}
		}
		// Suffix / prefix DP
		{
			{
				/*
				*/
			}
		}
		// Derived DP
		{
			{
				/*
				*/
			}

		}
		// Knapsack
		{
			{
				/*
				*/
			}

		}
		// Adhoc
		{
			{
				/*
				*/
			}

		}
		// DP optimized backtrack
		{
			{
				/*
				*/
			}

		}
		// Multiply DP
		{
			{
				/*
				*/
			}

		}
		// Breaking words
		{
			{
				/*
				*/


			}
		}
	}

	{
		/*
		*/
	}
