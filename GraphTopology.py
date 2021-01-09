import numpy as np
import networkx as nx




class GraphType():
    def __init__(self, type, n, p):
        self.type = type        # graph type
        self.n = n              # number of nodes
        self.p = p              # graph connectivity
        # self.W = np.zeros((self.n, self.n))

    def createGraph(self):
        conn = False
        G = None
        if self.type == 'erdos-renyi':
            while not conn:
                G = nx.erdos_renyi_graph(self.n, self.p)
                conn = nx.is_connected(G)
        if self.type == 'cycle':
            while not conn:
                G = nx.cycle_graph(self.n)
                conn = nx.is_connected(G)
        if self.type == 'expander':
            while not conn:
                G = nx.margulis_gabber_galil_graph(self.n)
                conn = nx.is_connected(G)
        if self.type == 'star':
            while not conn:
                G = nx.star_graph(self.n - 1)
                conn = nx.is_connected(G)
        self.G = G
        print(conn)
        self.create_weight_matrix_metropolis(G)
        return self.W

    def create_weight_matrix_metropolis(self, G):
        A = nx.to_numpy_matrix(G)
        # degree of the nodes
        D = np.sum(A, 1)
        D = np.array(D.transpose())
        D = D[0, :]
        D = D.astype(np.int64)
        self.W = np.zeros((A.shape))
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[1]):
                if A[i, j] != 0 and i != j:
                    self.W[i, j] = 1 / (max(D[i], D[j])+1)
                    self.W[j, i] = self.W[i, j]
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[1]):
                if i == j:
                    self.W[i, j] = 1 - np.sum(self.W[i, :])

    def ShowGraph(self):
        nx.draw(self.G)
