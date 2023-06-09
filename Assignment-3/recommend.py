import numpy as np
from scipy import sparse
from os import makedirs
from os.path import join
from collaborative import collaborative_filtering
from svd import svd
from cur import cur
import math

data_path = join('data', 'ml-1m', 'ratings.dat')
processed_files_dir = 'processed_data'

all_instances_file_name = 'ratings_all'
train_file_name = 'ratings_train'
test_file_name = 'ratings_test'

test_perc = 0.20        # percentage of testing instances

collaborative_neighbours = 150      # number of neighbours used in collaborative filtering
concepts = 40                       # number of concepts/eigen values to consider while performing SVD decomposition
CUR_no_cols = 4 * concepts          # number of columns and rows to select while performing CUR

txt_dir = join(processed_files_dir, 'txts')
sparse_dir = join(processed_files_dir, 'sparse_matrices')

makedirs(txt_dir, exist_ok=True)
makedirs(sparse_dir, exist_ok=True)


def get_sparse_path(name, normalized = False):
    """
    Returns the path of the sparse matrix given its name and normalized flag.

    Parameters:
    name (str): Name of the sparse matrix.
    normalized (bool): Flag indicating whether the matrix is normalized or not.

    Returns:
    str: Path of the sparse matrix.
    """
    suffix = '_normalized' if normalized else '_original'
    return join(sparse_dir, f'{name}{suffix}.npz')

def get_sparse_path_by_type(type, normalized = False):
    """
    Returns the path of the sparse matrix given its type and normalized flag.

    Parameters:
    type (str): Type of the sparse matrix ('train', 'test', or 'all').
    normalized (bool): Flag indicating whether the matrix is normalized or not.

    Returns:
    str: Path of the sparse matrix.
    
    Raises:
    ValueError: If an invalid value for the 'type' parameter is provided.
    """
    if type == 'train':
        return get_sparse_path(train_file_name, normalized)
    elif type == 'test':
        return get_sparse_path(test_file_name, normalized)
    elif type == 'all':
        return get_sparse_path(all_instances_file_name, normalized)
    else:
        raise ValueError('Invalid value for type')

def load_sparse_matrix(type, normalized = False):
    """
    Loads the sparse matrix given its type and normalized flag.

    Parameters:
    type (str): Type of the sparse matrix ('train', 'test', or 'all').
    normalized (bool): Flag indicating whether the matrix is normalized or not.

    Returns:
    scipy.sparse.csr_matrix: The loaded sparse matrix.
    """
    filepath = get_sparse_path_by_type(type, normalized)
    return sparse.load_npz(filepath)

def get_txt_path(name):
    """
    Returns the path of the text file given its name.

    Parameters:
    name (str): Name of the text file.

    Returns:
    str: Path of the text file.
    """
    return join(txt_dir, f'{name}.txt')

def get_txt_path_by_type(type):
    """
    Returns the path of the text file given its type.

    Parameters:
    type (str): Type of the sparse matrix ('train', 'test', or 'all').

    Returns:
    str: Path of the text file.
    
    Raises:
    ValueError: If an invalid value for the 'type' parameter is provided.
    """
    if type == 'train':
        return get_txt_path(train_file_name)
    elif type == 'test':
        return get_txt_path(test_file_name)
    elif type == 'all':
        return get_txt_path(all_instances_file_name)
    else:
        raise ValueError('Invalid value for type')

def rmse_spearman(matrix_predicted, matrix_actual, path):
    """
    Calculates the RMSE error and Spearman correlation between the predicted and actual matrices.

    Args:
        matrix_predicted (numpy.ndarray): The predicted matrix.
        matrix_actual (numpy.ndarray): The actual matrix.
        path (str): The path to the file containing the indices of the elements to be considered.

    Returns:
        None
    """

    total = 0.0
    no_instances = 0

    for line in open(path, 'r'):
        values = line.split(',')
        r, c = int(values[0]) - 1, int(values[1]) - 1
        total += math.pow((matrix_actual[r, c] - (matrix_predicted[r, c])), 2)
        no_instances += 1

    rho = 1 - (6 * total) / (no_instances * (math.pow(no_instances, 2) - 1))

    rmse, spear = math.sqrt(total) / no_instances, rho
    print(f'\nRMSE Error: {rmse}')
    print(f'Spearman Correlation: {spear * 100}%')


def precision_on_top_k(matrix_predicted, matrix_actual, k = 100):

    """
    Calculates the precision on top k based on the predicted and actual matrices.

    Args:
        matrix_predicted (numpy.ndarray): The predicted matrix.
        matrix_actual (numpy.ndarray): The actual matrix.
        k (int): The number of top elements to consider. Default value is 100.

    Returns:
        None
    """
    # create mask so that avg ratings are calculated over the same instances in both train and test matrix
    matrix_actual = matrix_actual.toarray()
    zero_values = (matrix_actual==0.0)
    matrix_predicted[zero_values] = 0

    # first find according to ratings in train matrix
    movie_mean_predicted = (np.squeeze(np.array(matrix_predicted.sum(axis = 0)/(matrix_predicted!=0).sum(axis = 0))))
    movie_mean_predicted[np.isnan(movie_mean_predicted)] = 0
    movie_mean_predicted_sorted = sorted(movie_mean_predicted.tolist(), reverse = True)[:k]

    # now find according to ratings in testing matrix
    movie_mean_actual = (np.squeeze(np.array(matrix_actual.sum(axis = 0)/(matrix_actual!=0).sum(axis = 0))))
    movie_mean_actual[np.isnan(movie_mean_actual)] = 0
    movie_mean_actual_sorted = sorted(movie_mean_actual.tolist(), reverse = True)[:k]

    # compare both lists to find precision
    fp = 0
    for i in range(k):
        fp += abs(movie_mean_predicted_sorted[i] - movie_mean_actual_sorted[i])
    fp = fp / k

    print(f'Precision on top {k}: {((k - fp) / k) * 100}%\n')


np.set_printoptions(precision = 5)
np.seterr(divide='ignore', invalid='ignore')


def main():
    """
    The main function that loads the matrices, performs collaborative filtering with and without baseline approach,
    performs SVD and CUR factorization, and prints out the RMSE error, Spearman correlation and precision on top k for
    each method.

    Args:
        None

    Returns:
        None
    """
    # load all matrices
    all_orig = load_sparse_matrix('all')
    all_norm = load_sparse_matrix('all', normalized=True)
    train_orig = load_sparse_matrix('train')
    train_norm = load_sparse_matrix('train', normalized=True)
    test_orig = load_sparse_matrix('test')
    test_norm = load_sparse_matrix('test', normalized=True)
    test_txt_path = get_txt_path_by_type('test')

    # # perform collaborative filtering with and without baseline approach
    collab_matrix = collaborative_filtering(train_norm, train_orig, test_orig, collaborative_neighbours)
    rmse_spearman(collab_matrix, test_orig, test_txt_path)
    precision_on_top_k(collab_matrix, all_orig)

    collab_matrix_baseline = collaborative_filtering(
    	train_norm, train_orig, test_orig, collaborative_neighbours, baseline=True
    )
    rmse_spearman(collab_matrix_baseline, all_orig, test_txt_path)
    precision_on_top_k(collab_matrix_baseline, all_orig)

    # # perform svd
    for energy in [1, 0.9]:
        svd_matrix = svd(train_norm, concepts, energy)
        rmse_spearman(svd_matrix, test_norm, test_txt_path)
        precision_on_top_k(svd_matrix, all_norm)

    # perform cur
    for energy in [1, 0.9]:
        cur_matrix = cur(train_norm, CUR_no_cols, concepts, energy)
        rmse_spearman(cur_matrix, test_norm, test_txt_path)
        precision_on_top_k(cur_matrix, all_norm)


if __name__ == '__main__':
    main()
