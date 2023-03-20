import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iters=10_000, tol=1e-6):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def __calc_distances(self, data):
        distances = np.zeros((data.shape[0], self.n_clusters))
        for idx, centroid in enumerate(self.centroids):
            distances[:, idx] = np.linalg.norm(data - centroid, axis=1)
        return distances

    def fit(self, data):
        n_samples, n_features = data.shape

        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = data[idx]

        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = self.__calc_distances(data)
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(data[self.labels == j], axis=0)

            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, data):
        distances = self.__calc_distances(data)
        return np.argmin(distances, axis=1)