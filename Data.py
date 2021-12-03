import numpy as np

class Data():
    def __init__(self, d, N, eigen_gap, K):
        self.d = d   # d = dimension of each sample
        self.N = N   # N = number of samples
        self.eigen_gap = eigen_gap    # eigengap of the covariance matrix
        self.K = K          # no. of eigenvectors to be estimated

    def generateSynthetic(self):
        A = np.random.rand(self.d, self.d)
        U, Sigma, V = np.linalg.svd(A)

        a = np.sqrt(np.linspace(1, 0.8, self.K))
        b = np.sqrt(np.linspace(0.8 * self.eigen_gap, 0.1, self.d - self.K))
        c = np.concatenate((a, b), axis=0)

        A_hat = U @ np.diag(c) * V.T


        # Z is a matrix of N standard normal vectors, size Nxd
        Z = np.random.multivariate_normal(np.zeros(self.d), np.identity(self.d), self.N)
        Z = Z.transpose()  # size dxN
        data = np.matmul(A_hat, Z)
        return data


    def computeTrueEV(self, data):
        N = data.shape[1]
        Cy = (1 / N) * np.dot(data, data.transpose())
        eigval_y, evd_y = np.linalg.eigh(Cy)
        eigval_y = np.flip(eigval_y)
        evd_y = np.fliplr(evd_y)
        X_gt = evd_y[:, 0:self.K]
        return X_gt