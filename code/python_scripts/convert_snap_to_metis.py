with open('soc-Epinions1.txt', 'r') as f:
    lines = [line.strip() for line in f if not line.startswith('#')]
V = max(max(int(u), int(v)) for u, v in (line.split() for line in lines)) + 1
E = len(lines)
with open('epinions.metis', 'w') as f:
    f.write(f"{V} {E} 1\n")
    adj = [[] for _ in range(V)]
    for u, v in (line.split() for line in lines):
        u, v = int(u) - 1, int(v) - 1
        adj[u].append((v, 1.0))
        adj[v].append((u, 1.0))
    for neighbors in adj:
        f.write(' '.join(f"{v+1} {w}" for v, w in neighbors) + '\n')
