import numpy as np
from collections import Counter
from tqdm import tqdm

def read_input(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        L = int(lines[0].split()[0])
        T = int(lines[1].split()[0])
        p0 = float(lines[2].split()[0])
        pk = float(lines[3].split()[0])
        dp = float(lines[4].split()[0])
    return L, T, p0, pk, dp


def generate_lattice(size, p):
    lattice = np.random.rand(size, size) < p
    return lattice.astype(int)


def burning_method(lattice):
    L = lattice.shape[0]
    visited = np.zeros_like(lattice, dtype=bool)
    stack = [(0, j) for j in range(L) if lattice[0, j] == 1]

    while stack:
        x, y = stack.pop()
        if x == L - 1:  
            return True
        if not visited[x, y]:
            visited[x, y] = True
            for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= nx < L and 0 <= ny < L and lattice[nx, ny] == 1 and not visited[nx, ny]:
                    stack.append((nx, ny))
    return False


def hoshen_kopelman(lattice):
    L = lattice.shape[0]
    labels = np.zeros_like(lattice, dtype=int)
    label = 1
    parent = {}

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]  
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i in range(L):
        for j in range(L):
            if lattice[i, j] == 1:
                neighbors = []
                if i > 0 and labels[i - 1, j] > 0: 
                    neighbors.append(labels[i - 1, j])
                if j > 0 and labels[i, j - 1] > 0:  
                    neighbors.append(labels[i, j - 1])

                if neighbors:
                    min_label = min(neighbors)
                    labels[i, j] = min_label
                    for n in neighbors:
                        union(min_label, n)
                else:
                    labels[i, j] = label
                    parent[label] = label
                    label += 1

    for i in range(L):
        for j in range(L):
            if labels[i, j] > 0:
                labels[i, j] = find(labels[i, j])

    cluster_sizes = Counter(labels.flatten())
    return labels, cluster_sizes


def monte_carlo_simulation(L, T, p0, pk, dp):
    p_values = np.arange(p0, pk + dp, dp)
    results = []

    for p in tqdm(p_values, desc="Overall Progress"):
        P_flow = 0
        smax_total = 0
        cluster_distribution = Counter()

        for _ in tqdm(range(T), desc=f"Simulating for p={p:.2f}", leave=False):
            lattice = generate_lattice(L, p)
            P_flow += burning_method(lattice)
            _, cluster_sizes = hoshen_kopelman(lattice)

            smax_total += max(cluster_sizes.values())

            cluster_distribution.update(cluster_sizes)

        P_flow /= T
        smax_avg = smax_total / T
        results.append((p, P_flow, smax_avg))

        dist_filename = f'Dist-p{p:.2f}L{L}T{T}.txt'
        with open(dist_filename, 'w') as file:
            for s, count in cluster_distribution.items():
                if s > 0:
                    file.write(f"{s}  {count}\n")

    results_filename = f'Ave-L{L}T{T}.txt'
    with open(results_filename, 'w') as file:
        for p, P_flow, smax_avg in results:
            file.write(f"{p}  {P_flow}  {smax_avg}\n")


def main():
    L, T, p0, pk, dp = read_input('perc-ini.txt')
    monte_carlo_simulation(L, T, p0, pk, dp)


if __name__ == "__main__":
    main()