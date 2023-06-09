import time
import numpy as np
from scipy.sparse import linalg
from scipy.sparse.linalg import LinearOperator


def svd_sparse(sparse_mat, num_singular_values):
    """
    Perform SVD on a sparse matrix using the eigsh method

    Parameters:
    sparse_mat : sparse matrix to be decomposed
    num_singular_values : number of singular values to retain

    Returns:
    U_large : left singular vectors
    sigma : singular values
    Vt_large : right singular vectors transposed
    """
    
    num_rows, num_cols = sparse_mat.shape

    XT_X = sparse_mat.T @ sparse_mat
    singular_values, left_singular_vectors = linalg.eigsh(XT_X, k=num_singular_values)
    singular_values = np.maximum(singular_values.real, 0)

    # in our case all singular values are going to be greater than zero
    # create sigma diagnol matrix
    s_large = np.sqrt(singular_values)
    sigma = np.zeros_like(singular_values)
    sigma[:num_singular_values] = s_large

    U_large = (sparse_mat @ left_singular_vectors) / s_large
    Vt_large = left_singular_vectors.T

    return U_large, sigma, Vt_large


def svd_retain_energy(sparse_mat, num_singular_values, energy=1):
    """
    Perform SVD on a sparse matrix and retain a certain percentage of the largest singular values

    Parameters:
    sparse_mat : sparse matrix to be decomposed
    num_singular_values : number of singular values to retain
    energy : percentage of energy to retain (default 1)

    Returns:
    U : left singular vectors
    Sigma : singular values
    Vt : right singular vectors transposed
    """

    U, Sigma, Vt = svd_sparse(sparse_mat, num_singular_values)
    sigma_squared_sum = np.square(Sigma).sum()  # sum of square of all singular values (diagonal elements in Sigma)

    i = 0
    while i < Sigma.shape[0] and np.square(Sigma[i:]).sum() < (energy * sigma_squared_sum):
        i += 1

    U = np.delete(U, np.s_[:i], 1)
    Sigma=Sigma[i:]
    Vt=np.delete(Vt, np.s_[:i], 0)

    return U, Sigma, Vt


def svd(sparse_mat, num_singular_values, energy=1):
    """
    Perform SVD Decomposition on the input sparse_mat
    Pass the copy of the sparse matrix to keep the original matrix unchanged

    Parameters:
    sparse_mat : input sparse_mat
    num_singular_values: number of largest singular values desired
    energy: retain energy% of largest singular values

    Returns : The dot product of U S and Vt matrix
    """

    start_time = time.time()
    print(f'SVD WITH {energy * 100}% ENERGY')

    U, Sigma, Vt = svd_retain_energy(sparse_mat, num_singular_values, energy)
    svd_matrix = U @ np.diag(Sigma) @ Vt

    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time:.2f} secs.")
    return svd_matrix
