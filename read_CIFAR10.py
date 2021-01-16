import numpy as np
import pickle
import math

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
M = np.mean(data, axis=1).reshape(d, 1)     #feature-wise mean
M_matrix = np.tile(M, (1, N))               #replicate the mean vector
data = (data - M_matrix)

Cy = (1 / N) * np.dot(data, data.transpose())
eigval_y, evd_y = np.linalg.eigh(Cy)
eigval_y = np.flip(eigval_y)
data = data/math.sqrt(eigval_y[0])

with open("Datasets/pickled/cifar10.pickle", 'wb') as handle:
    pickle.dump(data, handle)


