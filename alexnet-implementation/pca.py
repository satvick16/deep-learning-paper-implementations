import numpy as np

'''
X: dataset
M: principal components
S: covariance matrix

steps:
1. data normalization
2. find eigs for S
3. compute orthogonal projection matrix and project data onto subspace
'''


def normalize(X):
    N, D = X.shape
    mu = np.mean(X, axis=0)
    Xbar = X - mu

    return Xbar, mu


def eig(S):
    eigvals, eigvecs = np.linalg.eig(S)
    sort_indices = np.argsort(eigvals)[::-1]

    return eigvals[sort_indices], eigvecs[:, sort_indices]


def projection_matrix(B):
    return B @ np.linalg.inv(B.T @ B) @ B.T


def PCA(X, num_components):
    X_normalized, mean = normalize(X)
    S = np.cov(X, rowvar=False, bias=True)
    eig_vals, eig_vecs = eig(S)
    principal_vals, principal_components = eig_vals[:
                                                    num_components], eig_vecs[:, :num_components]
    principal_components = np.real(principal_components)
    reconst = (projection_matrix(principal_components)
               @ X_normalized.T).T + mean

    return reconst, mean, principal_vals, principal_components
