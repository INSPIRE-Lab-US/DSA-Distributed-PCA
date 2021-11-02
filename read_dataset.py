import numpy as np
import pickle
import gzip
import math

def read_data(dataset):
    if dataset == 'mnist':
        return read_mnist()
    elif dataset == 'cifar10':
        return read_cifar10()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_mnist():
    # Load the dataset
    (train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets) = pickle.load(
        gzip.open('Datasets/raw/mnist_py3k.pkl.gz', 'rb'))
    train_inputs = np.concatenate((train_inputs, valid_inputs))
    data = train_inputs.transpose()  # dimensionxnum_samples

    d = data.shape[0]
    N = data.shape[1]
    M = np.mean(data, axis=1).reshape(d, 1)  # feature-wise mean
    M_matrix = np.tile(M, (1, N))  # replicate the mean vector
    data = (data - M_matrix)

    return data

def read_cifar10():
    data_concatenated = []
    for i in range(1, 6):
        f = 'Datasets/raw/cifar-10-batches-py/data_batch_{}'.format(i)

        train_dict = unpickle(f)
        train_data = train_dict[b'data']
        if i == 1:
            data_concatenated = train_data
        else:
            data_concatenated = np.concatenate((data_concatenated, train_data))
    f = 'Datasets/raw/cifar-10-batches-py/test_batch'

    test_dict = unpickle(f)
    test_data = test_dict[b'data']
    data_concatenated = np.concatenate((data_concatenated, test_data))
    data_concatenated = data_concatenated[:, :1024]  # num_samplesxdimension
    data = data_concatenated.transpose()  # dimensionxnum_samples

    d = data.shape[0]
    N = data.shape[1]
    M = np.mean(data, axis=1).reshape(d, 1)  # feature-wise mean
    M_matrix = np.tile(M, (1, N))  # replicate the mean vector
    data = (data - M_matrix)

    Cy = (1 / N) * np.dot(data, data.transpose())
    eigval_y, evd_y = np.linalg.eigh(Cy)
    eigval_y = np.flip(eigval_y)
    data = data / math.sqrt(eigval_y[0])

    return data