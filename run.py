import numpy as np
from Algorithms import Algorithms
from GraphTopology import GraphType
from Data import Data

import pickle


# initialize variables
iterations = 10000
K = 20           # number of eigenvectors to be estimated

gtype = 'erdos-renyi'   # type of graph: erdos-renyi, cycle, star
num_nodes = 20          # number of nodes
p = 0.5                 # connectivity for erdos renyi graph
step_size = 0.7        # initial step size for DSA
step_sizeg = 0.8        # initial step size for GHA
step_sizep = 1        # initial step size for PGD
flag = 0                # flag = 0: constant step size, flag = 1: 1/t^0.2, flag = 2: 1/sqrt(t)

# generate graph
graphW = GraphType(gtype, num_nodes, p)
W = graphW.createGraph()
WW = np.kron(W, np.identity(K))

# import data set
dataset = 'cifar10'
test_data = Data(dataset)
with open("Datasets/pickled/{}.pickle".format(dataset), 'rb') as handle:
    data = pickle.load(handle)

# load EVD output
with open("Datasets/true_eigenvectors/EV_{}.pickle".format(dataset), 'rb') as f:
    X1 = pickle.load(f)
X_gt = X1[:, 0:K]

np.random.seed(1)
X_init = np.random.rand(data.shape[0], K)
X_init, r = np.linalg.qr(X_init)


test_run = Algorithms(data, iterations, K, num_nodes, initial_est=X_init, ground_truth=X_gt)

angle_sanger = test_run.centralized_sanger(step_size, flag)
angle_oi = test_run.OI()
angle_dsa = test_run.DSA(WW, step_size, flag)
angle_seqdistpm = test_run.seqdistPM(W, 50)
angle_dpgd = test_run.distProjGD(WW, step_sizep, flag)


with open('results/{}_K{}_stepsize{}_stepsizeg{}_stepsizep{}_flag{}_graphtype{}_n{}.pickle'.format(dataset, K, step_size, step_sizeg, step_sizep, flag, gtype, num_nodes), 'wb') as f:
    pickle.dump([angle_dsa, angle_sanger, angle_oi, angle_seqdistpm, angle_dpgd], f)
