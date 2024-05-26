import torch
import time
#import normal
import numpy as np
import random

INPUT_SEED = None

def random_seed():
    global INPUT_SEED
    if INPUT_SEED is None:
        return int(time.time())
    else:
        return INPUT_SEED

def set_seed(seed: int):
    global INPUT_SEED
    INPUT_SEED=seed
    random.seed(INPUT_SEED)
    np.random.seed(INPUT_SEED)
    torch.manual_seed(INPUT_SEED)

def avg_hypernode_density(edge_list):
    total_cardinality = 0
    num_edges = len(edge_list)
    for e in edge_list:
        total_cardinality += len(e)
    if num_edges == 0:
        return 0
    else:
        return total_cardinality / num_edges


def random_noise(dim, noise: float=0.2):
    if isinstance(dim, list):
        features = np.array(dim)
    elif isinstance(dim, torch.Tensor):
        features = dim.detach().cpu().numpy()
    elif not isinstance(dim, np.ndarray):
        raise TypeError("not supported")
    assert len(dim.shape) == 1, "dim not match"
    feature_set = np.unique(features).tolist()
    N, C = features.shape[0], len(feature_set)
    feature_list = []
    for i in range(N):
        feature_list.append(feature_set.index(features[i]))
    feature = np.array(feature_list)
    centers = np.zeros((N, C))
    centers[np.arange(N), features] = 1
    features = np.random.normal(centers, noise, size=(N, C))
    return torch.from_numpy(features).float()

def eigen_features(HG):
    L_sym = HG.L_sym # Symmetric Laplacian on HG 
    eigval, eigvecs = sp.linalg.eigs(L_sym.todense())
    top_eigenvals = eigenvals[::-1][:50]
    eigenvector_matrix = np.zeros((HG.num_v, 50))
    for i in range((HG.num_v)):
        node_eigenvecs = eigenvecs[:, ::-1][:, :50]
        eigenvector_matrix[i] = node_eigenvecs[i]
    
    return eigenvector_matrix
