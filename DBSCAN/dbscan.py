import csv
import math

# Load dataset
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data = [row for row in reader]
    return data

# Encode (for X,Y dataset)
def encode(row):
    return [float(row[1]), float(row[2])]

# Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Find neighbors
def region_query(X, point_idx, epsilon):
    neighbors = []
    for i in range(len(X)):
        if euclidean_distance(X[point_idx], X[i]) <= epsilon:
            neighbors.append(i)
    return neighbors

# Expand cluster
def expand_cluster(X, labels, point_types, point_idx, neighbors, cluster_id, epsilon, min_pts):
    labels[point_idx] = cluster_id
    point_types[point_idx] = "Core"

    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
            point_types[neighbor_idx] = "Border"

        elif labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(X, neighbor_idx, epsilon)

            if len(new_neighbors) >= min_pts:
                point_types[neighbor_idx] = "Core"
                for n in new_neighbors:
                    if n not in neighbors:
                        neighbors.append(n)
            else:
                point_types[neighbor_idx] = "Border"

        i += 1

# DBSCAN
def dbscan(X, epsilon, min_pts):
    labels = [0] * len(X)          # 0 = unvisited, -1 = noise
    point_types = [""] * len(X)    # Core / Border / Noise
    cluster_id = 0

    for i in range(len(X)):
        if labels[i] != 0:
            continue

        neighbors = region_query(X, i, epsilon)

        if len(neighbors) < min_pts:
            labels[i] = -1
            point_types[i] = "Noise"
        else:
            cluster_id += 1
            expand_cluster(X, labels, point_types, i, neighbors, cluster_id, epsilon, min_pts)

    return labels, point_types

# ================= RUN =================

raw_data = load_csv("dbscan.csv")
X = [encode(row) for row in raw_data]

epsilon = 1.0
min_pts = 3

labels, point_types = dbscan(X, epsilon, min_pts)

# Output
for i in range(len(X)):
    print(f"Point {i+1} {X[i]} → Cluster {labels[i]}, Type: {point_types[i]}")