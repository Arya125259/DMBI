import csv
import random
import math

# -------------------------
# Load CSV (numeric dataset only)
# -------------------------
def load_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            # convert all columns except ID to float
            data.append([float(x) for x in row[1:]])

    return data

# -------------------------
# Euclidean Distance
# -------------------------
def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# -------------------------
# Assign clusters
# -------------------------
def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]

    for point in data:
        distances = [distance(point, c) for c in centroids]
        idx = distances.index(min(distances))
        clusters[idx].append(point)

    return clusters

# -------------------------
# Update centroids
# -------------------------
def update_centroids(clusters):
    new_centroids = []

    for cluster in clusters:
        if len(cluster) == 0:
            continue

        centroid = [
            sum(dim) / len(cluster)
            for dim in zip(*cluster)
        ]
        new_centroids.append(centroid)

    return new_centroids

# -------------------------
# Convergence check
# -------------------------
def has_converged(old, new, eps=0.0001):
    if len(old) != len(new):
        return False

    for a, b in zip(old, new):
        for x, y in zip(a, b):
            if abs(x - y) > eps:
                return False
    return True

# -------------------------
# K-Means Algorithm
# -------------------------
def kmeans(data, k, max_iter=100):

    centroids = random.sample(data, k)

    for _ in range(max_iter):

        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

# -------------------------
# RUN PROGRAM
# -------------------------
filename = "Kmeans.csv"
data = load_csv(filename)

k = 3
centroids, clusters = kmeans(data, k)

print("\nFINAL CENTROIDS:")
for c in centroids:
    print(c)

print("\nCLUSTERS:")
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1}:")
    for point in cluster:
        print(point)