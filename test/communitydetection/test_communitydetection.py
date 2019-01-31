import teneto.communitydetection
import numpy as np


def test_community():
    np.random.seed(20)
    # Two graphlets templates with definite community structure
    a = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [
                 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    b = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [
                 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]])
    # Make into 3 time points
    G = np.stack([a, a, b, b]).transpose([1, 2, 0])
    C = teneto.communitydetection.temporal_louvain(G, intersliceweight=0.1, n_iter=1)
    assert C[0, 0] == C[1, 0] == C[2, 0]
    assert C[3, 0] == C[4, 0] == C[5, 0]
    assert C[0, 2] == C[1, 2] == C[2, 2] == C[3, 2]
    assert C[4, 2] == C[5, 2]
    assert C[3, 0] != C[0, 0]
    assert C[4, 2] != C[0, 2]

