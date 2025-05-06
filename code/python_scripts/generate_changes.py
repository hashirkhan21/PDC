import random
V = 75879  # Number of vertices in soc-Epinions1
with open('changes_large.txt', 'w') as f:
    for i in range(5000):
        u = random.randint(1, V)
        v = random.randint(1, V)
        w = random.uniform(0.1, 10.0)
        f.write(f"a {u} {v} {w:.1f}\n")
    for i in range(5000):
        u = random.randint(1, V)
        v = random.randint(1, V)
        f.write(f"d {u} {v}\n")
