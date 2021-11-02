import numpy as np
from Algorithms import Algorithms
from GraphTopology import GraphType
from Data import Data
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--monte_trial", help="A number to indicate how many Monte Carlo trials to run, default value is 10", type=int, default=10)
parser.add_argument("-d","--dimension", help="Dimension of the data samples, default value is 20", type=int, default=20)
parser.add_argument("-K", "--K", help="number of eigenvectors to be estimated, default number is 5", type = int, default=5)
parser.add_argument("-EG","--eigengap", help="eigengap between Kth and (K+1)th eigenvalues", type = float, default=0.6)
parser.add_argument("-gt","--graphtype", help="graph topology, default topology is erdos-renyi", choices=['erdos-renyi', 'star', 'cycle'], type = str, default='erdos-renyi')
parser.add_argument("-n", "--num_nodes", help="number of nodes in the network, default number is 10", type = int, default=10)
parser.add_argument("-s", "--stepsize", help="step size (or learning rate) for DSA and centralized GHA algorithms, default value is 0.1", type = float, default=0.1)



args = parser.parse_args()
# initialize variables
iterations = 10000
N = 10000       # number of data samples

d = args.dimension          # dimension of data samples
K = args.K                  # number of eigenvectors to be estimated
eigengap = args.eigengap     # eigen gap between K+1 and Kth eigenvalue

gtype = args.graphtype   # type of graph: erdos-renyi, cycle, star
p = 0.5                 # connectivity for erdos renyi graph

num_nodes = args.num_nodes      # number of nodes
step_size = args.stepsize       # initial step size for DSA
step_sizeg = args.stepsize      # initial step-size for GHA
step_sizep = 0.1                # initial step size for PGD
flag = 0                        # flag = 0: constant step size, flag = 1: 1/t^0.2, flag = 2: 1/sqrt(t)

# generate graph
graphW = GraphType(gtype, num_nodes, p)
W = graphW.createGraph()
WW = np.kron(W, np.identity(K))

# Monte Carlo simulations
MonteCarlo = args.monte_trial
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
    print(X_gt.shape)
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

