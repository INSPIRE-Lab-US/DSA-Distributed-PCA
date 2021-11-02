import numpy as np
from Algorithms import Algorithms
from GraphTopology import GraphType
import pickle
import argparse
import read_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-K", "--K", help="number of eigenvectors to be estimated, default number is 5", type = int, default=5)
parser.add_argument("-n", "--num_nodes", help="number of nodes in the network, default number is 10", type = int, default=10)
parser.add_argument("-s", "--stepsize", help="step size (or learning rate) for DSA and centralized GHA algorithms, default value is 0.1", type = float, default=0.1)
parser.add_argument("-ds", "--dataset", help="dataset used for the experiment, default is MNIST",
                   choices=['mnist', 'cifar10'], type = str, default="mnist")
args = parser.parse_args()

# initialize variables
iterations = 10000

K = args.K              # number of eigenvectors to be estimated

gtype = 'erdos-renyi'
p = 0.5                 # connectivity for erdos renyi graph

num_nodes = args.num_nodes           # number of nodes
step_size = args.stepsize            # initial step size for DSA
step_sizeg = args.stepsize           # initial step size for GHA
step_sizep = 1                       # initial step size for PGD
flag = 0                             # flag = 0: constant step size, flag = 1: 1/t^0.2, flag = 2: 1/sqrt(t)

# generate graph
graphW = GraphType(gtype, num_nodes, p)
W = graphW.createGraph()
WW = np.kron(W, np.identity(K))

# import data set
dataset = args.dataset
data = read_dataset.read_data(dataset)

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
