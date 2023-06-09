import time
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

def collaborative_filtering(sparse_matrix, sparse_matrix_original, sparse_matrix_test_original, k, baseline = False):  
    """
    Perform collaborative filtering on the sparse_matrix being sent.
    
    Args:
    sparse_matrix (numpy.ndarray): Sparse matrix on which collaborative filtering needs to be performed.
    sparse_matrix_original (numpy.ndarray): Original sparse matrix.
    sparse_matrix_test_original (numpy.ndarray): Original test sparse matrix.
    k (int): Number of similar users to consider for each user.
    baseline (bool, optional): Whether to use baseline approach. Default is False.
    
    Returns:
    numpy.ndarray: Collaborative matrix after performing collaborative filtering.
    """
    
    start = time.time()

    if baseline==False:
        print(f'COLLABORATIVE FILTERING WITHOUT BASELINE APPROACH')
    else:
        print(f'COLLABORATIVE FILTERING WITH BASELINE APPROACH')

    collaborative_matrix = sparse_matrix_original.toarray()

    # store rows and norm(row)
    row_norm = []
    r = 0
    while r < sparse_matrix.shape[0]:
        row_norm.append(linalg.norm(sparse_matrix.getrow(r)))
        r += 1
    
    row_norm = np.reshape(np.array(row_norm), (sparse_matrix.shape[0], 1))

    # store product of norm every 2 rows in users in row_norm_products
    row_norm_products = np.matmul(row_norm, row_norm.T)

    # store dot product of every 2 rows in dot_products
    dot_products = sparse_matrix @ sparse_matrix.T.A

    # get indices of upper triangular matrix
    aaa=dot_products.shape[0]
    iu1 = np.triu_indices(aaa, 1)
    colu_indices,rowu_indices = iu1

    # store similarities for each user-user tuple
    similarity = np.full((sparse_matrix.shape[0], sparse_matrix.shape[0]), -2, dtype=np.float32)
    np.fill_diagonal(similarity, 0)

    valid_indices = np.logical_and(~np.isnan(dot_products), ~np.isnan(row_norm_products))
    similarity[valid_indices] = dot_products[valid_indices] / row_norm_products[valid_indices]

    simm=np.nan_to_num(similarity, nan=0.0)
    similarity = simm

    # copy upper triangular values to lower triangle
    similarity.T[rowu_indices, colu_indices] = similarity[rowu_indices, colu_indices]

    # store indices of closest k users for every user
    neighbourhood = np.zeros((sparse_matrix.shape[0], k), dtype = np.int32)
    i = 0
    bbb=sparse_matrix.shape[0]
    while i < bbb:
        neighbourhood[i] = similarity[i,:].argsort()[-k:][::-1]
        i += 1

    # store similarity of user u with its k neighbours
    similarity_user_neighbour = np.zeros((similarity.shape[0], neighbourhood.shape[1]), dtype=np.float32)

    for i in range(similarity.shape[0]):
        similarity_user_neighbour[i] = similarity[i, neighbourhood[i]]

    row_test_indices, col_test_indices = sparse_matrix_test_original.nonzero()

    testt=5
    if baseline==False:
        testt+=1
        for i, (r, c) in enumerate(zip(row_test_indices, col_test_indices)):
            collaborative_matrix[r, c] = np.dot(similarity_user_neighbour[r], np.array([collaborative_matrix[j,c] for j in neighbourhood[r]]))/(similarity_user_neighbour[r].sum())
            testt+=1
    else:
        # find mean of whole original sparse matrix
        non_zero_values = sparse_matrix_original[sparse_matrix_original != 0]
        mean = np.mean(non_zero_values)
        testt=1
        # find mean rating for every user
        row_sums = sparse_matrix_original.sum(axis=1)
        nonzero_counts = (sparse_matrix_original != 0).sum(axis=1)
        user_mean = np.squeeze(np.array(row_sums / nonzero_counts))
        user_mean = user_mean.tolist()
        # find mean rating of every movie
        row_sums = sparse_matrix_original.sum(axis=0)
        nonzero_counts = (sparse_matrix_original != 0).sum(axis=0)
        movie_mean = np.squeeze(np.array(row_sums / nonzero_counts))
        movie_mean = movie_mean.tolist()

        row_indices, col_indices = sparse_matrix_original.nonzero()
        data = np.array([user_mean[r] + movie_mean[c] - mean for (r,c) in zip(row_indices, col_indices)])
        testt=2
        shape = sparse_matrix_original.shape
        baseline_matrix = sparse.coo_matrix((data, (row_indices,col_indices)),shape).toarray()

        for i, (r, c) in enumerate(zip(row_test_indices, col_test_indices)):
            basee=baseline_matrix[r,c]
            collaborative_matrix[r, c] = basee + np.dot(similarity_user_neighbour[r], np.array([collaborative_matrix[j,c]-baseline_matrix[j,c] for j in neighbourhood[r]]))/(similarity_user_neighbour[r].sum())

    print('Total time taken was ' + '{0:.2f}'.format(time.time() - start) + ' secs.')
    collaborative_matrix[np.isnan(collaborative_matrix)] = 0.0
    test2=0
    return collaborative_matrix
