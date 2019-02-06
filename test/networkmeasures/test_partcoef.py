import numpy as np
import teneto


def test_partcoef():

    communities = [1, 1, 1, 0]
    G = np.zeros([4, 4, 3])
    G[0, 1, [0, 2]] = 1
    G[2, 3, [0, 2]] = 1
    G[1, 2, [1, 2]] = 1
    G[0, 3, [1, 2]] = 1
    G += G.transpose([1, 0, 2])
    G[1:, 1:, 0] = 0.5
    part = teneto.networkmeasures.temporal_participation_coeff(
        G, np.array(communities))
    # Calculate value of node 1 at time 0.
    assert part[1, 0] == 1-(np.power(1.5/2, 2)+np.power(.5/2, 2))
    # Hardcode known partcoeff
    assert np.all(part[:, 2] == [0.5, 0, 0.5, 0])
