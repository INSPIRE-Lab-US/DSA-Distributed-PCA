# DistributedPCA_DSA : Codebase for DSA
## General Information

This repository contains implementations of **Distributed Sanger's Algorithm (DSA)** introduced in the paper [A Linearly Convergent Algorithm for Dsitributed Principal Component Analysis](https://arxiv.org/pdf/2101.01300.pdf). This paper proposed a novel Hebbian based learning algorithm for learning the eigenvectors of a covariance matrix when data is distributed across an arbitrary network of nodes.

## Citation

If you use any part of this code, then please cite the following paper: Gang A., Bajwa W.U., "A Linearly Convergent Algorithm for Distributed Principal Component Analysis", arXiv preprint arXiv:2101.01300. Jan. 2021.

## Summary of Experiments

The codebase implements DSA and compares with 4 baseline algorithms namely centralized GHA (proposed by Sanger), centralized orthogonal iterations (OI), sequential distributed power method (SeqDistPM) and distributed projected gradient descent (DPGD). 

## Data

Similar to the results in the paper, the codes are provided to generate results on two kinds of data:

1.  Synthetic data -- Synthetic data of certain dimension and eigengap is generated and results are obtained for all the 5 methods mentioned above. 
2. Real world data -- MNIST and CIFAR10 datasets were used in our experiments. The MNIST data is available from this [website](http://yann.lecun.com/exdb/mnist/), while cifar10 data is available [here](https://www.cs.toronto.edu/~kriz/cifar.html). Both these datasets were download, read into numpy arrays and then pickled for use in the experiments. The pickled arrays are available in `./Datasets/pickled`. The eigenvalue decomposition for the covariance matrices of both the datasets is also available in `./Datasets/true_eigenvectors` and can be readily loaded for measuring the performance accuracy of the algorithms. If `K` eigenvectors are being estimated, just use first `K` columns from the loaded EVs. 

=======
