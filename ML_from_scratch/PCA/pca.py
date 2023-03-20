import numpy as np

class PCA:

    def __init__(self, n_components):
        self._n_components = n_components
        self._components = None
        self._mean = None 

    def fit(self, X):
        # center the data
        self._mean = np.mean(X, axis=0)
        X = X - self._mean

        # compute the covariance matrix
        cov = np.cov(X, rowvar=False)

        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors as the principal components
        self._components = eigenvectors[:, :self._components]

    def transform(self, X):
        # center the data
        # self._mean = np.mean(X, axis=0)
        X = X - self._mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self._components)

        return X_transformed