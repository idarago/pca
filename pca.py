'''
Implementation of the Principal Component Analysis algorithm
------------------------------------------------------------
Principal Component Analysis (PCA) is a dimension-reduction algorithm.
The idea is to use the singular value decomposition of a data matrix
to obtain the directions that explain the most of the variance.
This boils down to finding the eigenvalue decomposition of the covariance matrix.
'''
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.cov = None
        self.explained_variance = None
        self.singular_values = None
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Unbiased estimator of the covariance
        self.cov = X.T @ X / (len(X)-1)
        
        # Eigenvalues, eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)

        # Sort eigenvectors and choose the most relevant ones
        sorted_eigenvectors_eigenvalues = list(zip(*sorted(zip(eigenvectors.T,eigenvalues), key = lambda x : x[1], reverse=True)))
        self.singular_values = sorted_eigenvectors_eigenvalues[1]
        self.explained_variance = sum(s**2 / sum(s**2 for s in self.singular_values) for s in self.singular_values[:self.n_components])
        self.components = np.array(sorted_eigenvectors_eigenvalues[0][:self.n_components])

    def transform(self, X):
        # Apply dimensionality reduction to X
        return np.dot(X - self.mean, self.components.T)

    def inverse_transform(self, y):
        # Transform data back to its original space
        return np.dot(y, self.components) + self.mean
