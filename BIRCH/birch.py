import numpy as np
import pandas as pd


# ---------------- CF STRUCTURE ----------------
class CF:
    def __init__(self, dim):
        self.N = 0
        self.LS = np.zeros(dim, dtype=float)
        self.SS = np.zeros(dim, dtype=float)
        self.points = []

    def add(self, x):
        x = np.array(x, dtype=float)
        self.N += 1
        self.LS += x
        self.SS += x ** 2
        self.points.append(x)

    def centroid(self):
        return self.LS / self.N


# ---------------- BIRCH ----------------
class BIRCH:
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        self.clusters = []
        self.dim = None

    # -------- LOAD CSV --------
    def load_csv(self, file_path):
        df = pd.read_csv(file_path)

        # keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        data = df.values

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return data

    # -------- FIT MODEL --------
    def fit(self, data):
        data = np.array(data, dtype=float)

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        self.dim = data.shape[1]

        print("\n=== INSERTION PROCESS ===")

        for x in data:
            self._insert(x)

    # -------- INSERT POINT --------
    def _insert(self, x):
        best = None
        min_dist = float("inf")

        # find nearest cluster
        for c in self.clusters:
            dist = np.linalg.norm(c.centroid() - x)
            if dist < min_dist:
                min_dist = dist
                best = c

        # first cluster
        if best is None:
            cf = CF(self.dim)
            cf.add(x)
            self.clusters.append(cf)

            print(f"\nInsert {x.tolist()} → NEW Cluster 1")
            print("Centroid:", np.round(cf.centroid(), 2))
            return

        idx = self.clusters.index(best) + 1

        # create temp CF for safe update
        temp = CF(self.dim)
        temp.N = best.N
        temp.LS = best.LS.copy()
        temp.SS = best.SS.copy()
        temp.add(x)

        # threshold decision (centroid distance based)
        if min_dist <= self.threshold:
            best.add(x)
            print(f"\nInsert {x.tolist()} → added to Cluster {idx}")
            print("Updated Centroid:", np.round(best.centroid(), 2))
        else:
            cf = CF(self.dim)
            cf.add(x)
            self.clusters.append(cf)

            print(f"\nInsert {x.tolist()} → NEW Cluster {len(self.clusters)}")
            print("Centroid:", np.round(cf.centroid(), 2))

    # -------- SHOW FINAL OUTPUT --------
    def show(self):
        print("\n\n=== FINAL CLUSTERS ===")

        for i, c in enumerate(self.clusters):
            print(f"\nCluster {i+1}")
            print("Points:", np.array(c.points).tolist())
            print("Centroid:", np.round(c.centroid(), 2))


# ---------------- RUN ----------------
if __name__ == "__main__":

    model = BIRCH(threshold=5.0)

    file_path = "dataset.csv"   # <-- your CSV file

    data = model.load_csv(file_path)
    model.fit(data)
    model.show()