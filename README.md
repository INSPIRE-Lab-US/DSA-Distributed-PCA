# DistributedPCA_DSA : Codebase for DSA
## General Information

This repository contains implementations of **Distributed Sanger's Algorithm (DSA)** introduced in the paper [A Linearly Convergent Algorithm for Dsitributed Principal Component Analysis](https://arxiv.org/pdf/2101.01300.pdf). This paper proposed a novel Hebbian based learning algorithm for learning the eigenvectors of a covariance matrix when data is distributed across an arbitrary network of nodes.

## Citation

The code in this repo is being released under the GNU General Public License v3.0; please refer to the [LICENSE](https://github.com/INSPIRE-Lab-US/DistributedPCA_DSA/blob/master/LICENSE) file in the repo for detailed legalese pertaining to the license. If you use any part of this code, then please cite the original paper and this codebase: 

###### Paper Citation

Gang A., Bajwa W.U., "A Linearly Convergent Algorithm for Distributed Principal Component Analysis", arXiv preprint arXiv:2101.01300. Jan. 2021.

###### Codebase citation

## Summary of Experiments

The codebase implements DSA and compares with 4 baseline algorithms namely centralized GHA (proposed by Sanger), centralized orthogonal iterations (OI), sequential distributed power method (SeqDistPM) and distributed projected gradient descent (DPGD). 

## Data

Similar to the results in the paper, the codes are provided to generate results on two kinds of data:

1.  Synthetic data -- Synthetic data of certain dimension and eigengap is generated and results are obtained for all the 5 methods mentioned above. 
2. Real world data -- MNIST and CIFAR10 datasets were used in our experiments. The MNIST data is available from this [website](http://yann.lecun.com/exdb/mnist/), while cifar10 data is available [here](https://www.cs.toronto.edu/~kriz/cifar.html). The raw datasets are provided in `./Datasets/raw` folders. The `read_MNIST.py` and `read_CIFAR10.py` scripts when run will read the raw data, convert them to numpy arrays, pickle them and store those in `./Datasets/pickled` folder to be used later for experiments. The eigenvalue decomposition for the covariance matrices of both the datasets is available in `./Datasets/true_eigenvectors` and can be readily loaded for measuring the performance accuracy of the algorithms. If `K` eigenvectors are being estimated, just use first `K` columns from the loaded EVs. 

## Summary of Code

The  `run_synthetic.py` and `run.py` are the main driving scripts for synthetic and real world data experiments respectively. The necessary functions required for the experiments are called within this functions and all the necessary parameters like number of eigenvectors to be estimated `K` , number of nodes in the network `num_nodes` etc. can be initialized in these scripts. In case of real world data, remember to first use the `read_MNIST.py` or `read_CIFAR10.py` (depending on the dataset being used) to read the raw data and convert them into formats usable by the codes.

## Requirements and Dependencies

The code is written in Python. To reproduce the environment with necessary dependencies needed for running of the code in this repo, we recommend that the users create a `conda` environment using the `environment.yml` YAML file that is provided in the repo. Assuming the conda management system is installed on the user's system, this can be done using the following: 

``` $ conda env create -f environment.yml ```

In the case users don't have conda installed on their system, they should check out the `environment.yml` file for the appropriate version of Python as well as the necessary dependencies with their respective versions needed to run the code in the repo.

## Contributors

The algorithmic implementations, experiments and reproduciblity of these codes was done by:

1. [Arpita Gang](https://www.linkedin.com/in/arpitagang/)
2. [Waheed U. Bajwa](http://www.inspirelab.us/) 







