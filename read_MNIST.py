import numpy as np
import pickle
import gzip



# Load the dataset
(train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets) = pickle.load(
    gzip.open('Datasets/raw/mnist_py3k.pkl.gz', 'rb'))
train_inputs = np.concatenate((train_inputs, valid_inputs))
data = train_inputs.transpose()  # dimensionxnum_samples

d = data.shape[0]
N = data.shape[1]
M = np.mean(data, axis=1).reshape(d, 1)     #feature-wise mean
M_matrix = np.tile(M, (1, N))               #replicate the mean vector
data = (data - M_matrix)

with open("Datasets/pickled/mnist.pickle", 'wb') as handle:
    pickle.dump(data, handle)
