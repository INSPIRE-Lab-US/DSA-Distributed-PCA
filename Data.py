import numpy as np
import pickle
import gzip

class Data():
    def __init__(self, dataset):
        self.dataset = dataset

    def generateSynthetic(self, d, N, eigen_gap, K):
        # d = dimension of each sample
        # N = number of samples
        a = np.linspace(1, 0.8, K)
        b = np.linspace(0.8 * eigen_gap, 0.1, d - K)
        c = np.concatenate((a, b), axis=0)
        Cov = np.diag(c)
        A = np.linalg.cholesky(Cov)
        # A = np.random.randn(d, d)

        # Z is a matrix of N standard normal vectors, size Nxd
        Z = np.random.multivariate_normal(np.zeros(d), np.identity(d), N)
        Z = Z.transpose()  # size dxN
        data = np.matmul(A, Z)
        return data

    def centeredData(self, data):
        d = data.shape[0]
        N = data.shape[1]
        M = np.mean(data, axis=1).reshape(d, 1)     #feature-wise mean
        M_matrix = np.tile(M, (1, N))               #replicate the mean vector
        return (data - M_matrix)

    def load_MNIST(self):
        # Load the dataset
        (train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets) = pickle.load(gzip.open('Datasets/mnist_py3k.pkl.gz', 'rb'))
        train_inputs = np.concatenate((train_inputs, valid_inputs))
        data = train_inputs.transpose()        #dimensionxnum_samples
        data = self.centeredData(data)         # zero mean data
        return data

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_CIFAR10(self):
        data_concatenated = []
        for i in range(1, 6):
            f = 'Datasets/cifar-10-batches-py/data_batch_{}'.format(i)

            train_dict = self.unpickle(f)
            train_data = train_dict[b'data']
            if i == 1:
                data_concatenated = train_data
            else:
                data_concatenated = np.concatenate((data_concatenated, train_data))
        f = 'Datasets/cifar-10-batches-py/test_batch'

        test_dict = self.unpickle(f)
        test_data = test_dict[b'data']
        data_concatenated = np.concatenate((data_concatenated, test_data))
        data_concatenated = data_concatenated[:, :1024]         #num_samplesxdimension
        data = data_concatenated.transpose()                    #dimensionxnum_samples
        data = self.centeredData(data)                          # zero-mean data
        return data

    def computeTrueEV(self, data, K):
        N = data.shape[1]
        Cy = (1 / N) * np.dot(data, data.transpose())
        eigval_y, evd_y = np.linalg.eigh(Cy)
        eigval_y = np.flip(eigval_y)
        evd_y = np.fliplr(evd_y)
        X_gt = evd_y[:, 0:K]
        print(eigval_y)
        # print(eigval_y[K]/eigval_y[K-1])
        return X_gt