import numpy as np
import teneto
from teneto.temporalcommunity import allegiance

def test_allegiance():

    np.random.seed(10)
    # Two graphlets templates with definite community structure
    a = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [
                 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    b = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [
                 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]])
    # Make into 4 time points
    G = np.stack([a, a, b, b]).transpose([1, 2, 0])

    # returns node,time array of community assignments
    community = teneto.communitydetection.temporal_louvain(
        G, intersliceweight=0.1, n_iter=1)

    P = allegiance(community)
    # Answers are handcoded based on analytic truth
    if not P[0,1] == P[0,2] == P[1,0] == P[1,2] == P[2,0] == P[2,1] == P[4,5] == P[5,4] == 1.0:
        raise AssertionError()
    if not P[0,3] == P[1,3] == P[2,3] == P[4,3] == P[5,3] == 0.5:
        raise AssertionError()
    if not P[3,0] == P[3,1] == P[3,2] == P[3,4] == P[3,5] == 0.5:
        raise AssertionError()
