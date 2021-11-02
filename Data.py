import numpy as np

class Data():
    def __init__(self, d, N, eigen_gap, K):
        self.d = d   # d = dimension of each sample
        self.N = N   # N = number of samples
        self.eigen_gap = eigen_gap    # eigengap of the covariance matrix
        self.K = K          # no. of eigenvectors to be estimated

    def generateSynthetic(self):
        a = np.linspace(1, 0.8, self.K)
        b = np.linspace(0.8 * self.eigen_gap, 0.1, self.d - self.K)
        c = np.concatenate((a, b), axis=0)
        Cov = np.diag(c)
        A = np.linalg.cholesky(Cov)
        # A = np.random.randn(d, d)

        # Z is a matrix of N standard normal vectors, size Nxd
        Z = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d), self.N)
        Z = Z.transpose()  # size dxN
        data = np.matmul(A, Z)
        return data


    def computeTrueEV(self, data):
        N = data.shape[1]
        Cy = (1 / N) * np.dot(data, data.transpose())
        eigval_y, evd_y = np.linalg.eigh(Cy)
        eigval_y = np.flip(eigval_y)
        evd_y = np.fliplr(evd_y)
        X_gt = evd_y[:, 0:self.K]
        return X_gt