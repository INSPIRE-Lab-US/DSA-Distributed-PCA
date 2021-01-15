import numpy as np
import matplotlib.pyplot as plt
from Algorithms import Algorithms
from GraphTopology import GraphType
from Data import Data

import pickle


# initialize variables
iterations = 10000
N = 10000       # number of data samples
d = 20          # dimension of data samples
K = 2           # number of eigenvectors to be estimated
eigengap = 0.8  # eigen gap between K+1 and Kth eigenvalue

gtype = 'erdos-renyi'   # type of graph: erdos-renyi, cycle, star
num_nodes = 10          # number of nodes
p = 0.5                 # connectivity for erdos renyi graph
step_size = 0.5        # initial step size for DSA
step_sizeg = 0.5        # initial step-size for GHA
step_sizep = 0.5       # initial step size for PGD
flag = 0                # flag = 0: constant step size, flag = 1: 1/t^0.2, flag = 2: 1/sqrt(t)

# generate graph
graphW = GraphType(gtype, num_nodes, p)
W = graphW.createGraph()
WW = np.kron(W, np.identity(K))

# Monte Carlo simulations
MonteCarlo = 10
angle_oi = np.zeros((MonteCarlo,), dtype=np.object)
angle_sanger = np.zeros((MonteCarlo,), dtype=np.object)
angle_dsa = np.zeros((MonteCarlo,), dtype=np.object)
angle_seqdistpm = np.zeros((MonteCarlo,), dtype=np.object)
angle_dpgd = np.zeros((MonteCarlo,), dtype=np.object)
dataset = 'synthetic'
for m in range(MonteCarlo):
    # generate synthetic data
    test_data = Data(d, N, eigengap, K)

    np.random.seed(10+m)
    data = test_data.generateSynthetic()
    X_gt = test_data.computeTrueEV(data)


    # initial estimate
    X_init = np.random.rand(data.shape[0], K)
    X_init, r = np.linalg.qr(X_init)

    # run algorithms on the data
    test_run = Algorithms(data, iterations, K, num_nodes, initial_est=X_init, ground_truth=X_gt)

    angle_sanger[m] = test_run.centralized_sanger(step_sizeg, flag)
    angle_oi[m] = test_run.OI()
    angle_dsa[m] = test_run.DSA(WW, step_size, flag)
    angle_seqdistpm[m] = test_run.seqdistPM(W, 50)
    angle_dpgd[m] = test_run.distProjGD(WW, step_sizep, flag)

with open('results/{}_d{}_K{}_eigengap{}_stepsize{}_stepsizeg{}_stepsizep{}_flag{}_graphtype{}_n{}.pickle'.format(dataset, d,K, eigengap, step_size, step_sizeg, step_sizep, flag, gtype,  num_nodes), 'wb') as f:
    pickle.dump([angle_dsa, angle_sanger, angle_oi, angle_seqdistpm, angle_dpgd], f)

