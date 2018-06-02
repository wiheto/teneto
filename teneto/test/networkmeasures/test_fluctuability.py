
import teneto
import numpy as np


def test_fluct():
    # Test volatility
    G = np.zeros([3, 3, 3])
    G[0, 1, [0, 1, 2]] = 1
    G[0, 2, 1] = 1
    G[1, 2, 2] = 1
    G = G + G.transpose([1, 0, 2])
    fluct = teneto.networkmeasures.fluctuability(G)
    # Hardcorde answer
    assert fluct == 5/3
