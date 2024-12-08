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
    labels = np.zeros_like(lattice, dtype=int)
    t = 2

    for j in range(L):
        if lattice[0, j] == 1:
            labels[0, j] = t

    while True:
        new_burning = False
        for i in range(L):
            for j in range(L):
                if labels[i, j] == t:
                    for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= ni < L and 0 <= nj < L and lattice[ni, nj] == 1 and labels[ni, nj] == 0:
                            labels[ni, nj] = t + 1
                            new_burning = True
                            if ni == L - 1:
                                return True
        if not new_burning:
            break
        t += 1

    return False


def hoshen_kopelman(lattice, update_labels=False):
    L = lattice.shape[0]
    labels = np.zeros_like(lattice, dtype=int)
    label_map = {}
    current_label = 2  # Start labeling from 2 to differentiate from binary 0/1 input

    def find_root(label):
        # Find the root label (handle negative references)
        root = label
        while label_map[root] < 0:
            root = -label_map[root]
        return root


    for i in range(L):
        for j in range(L):
            if lattice[i, j] == 1:  # Process only occupied sites
                neighbors = []

                # Get top and left neighbors
                if i > 0 and labels[i - 1, j] > 0:
                    neighbors.append(labels[i - 1, j])
                if j > 0 and labels[i, j - 1] > 0:
                    neighbors.append(labels[i, j - 1])

                if not neighbors:  # No neighbors
                    labels[i, j] = current_label
                    label_map[current_label] = 1  # Assign a new label and initialize cluster size
                    current_label += 1
                else:
                    root_labels = [find_root(n) for n in neighbors]
                    primary_label = min(root_labels)
                    labels[i, j] = primary_label

                    # Update cluster size
                    label_map[primary_label] += 1

                    # Merge clusters if necessary
                    for root_label in root_labels:
                        if root_label != primary_label:
                            label_map[primary_label] += label_map[root_label]
                            label_map[root_label] = -primary_label

    # Update labels to resolve references (optional step)
    if update_labels:
        for i in range(L):
            for j in range(L):
                if labels[i, j] > 0:
                    labels[i, j] = find_root(labels[i, j])

    # Collect cluster sizes
    cluster_sizes = Counter(
        size for size in label_map.values() if size > 0
    )

    return labels, cluster_sizes


def monte_carlo_simulation(L, T, p0, pk, dp):
    p_values = np.arange(p0, pk, dp)
    results = []

    for p in tqdm(p_values, desc="Overall Progress"):
        P_flow = 0
        smax_total = 0
        cluster_distribution = Counter()

        for _ in tqdm(range(T), desc=f"Simulating for p={p:.2f}", leave=False):
            lattice = generate_lattice(L, p)
            P_flow += burning_method(lattice)
            _, cluster_sizes = hoshen_kopelman(lattice)

            smax_total += max(cluster_sizes.keys())

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