import teneto
import numpy as np
import pytest


def test_topooverlap():
    # Define test data
    G = np.array(
        [[[0., 0., 0., 0.],
        [1., 1., 1., 0.],
        [0., 0., 1., 0.]],
        [[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 1., 1.]],
        [[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]])
    topoover = teneto.networkmeasures.topological_overlap(G)
    # Given the above, the following should be true
    # All edges persit
    assert topoover[0, 0] == 1
    # No edges persis
    assert topoover[1, 0] == 0
    # When one of two edges persists
    assert topoover[0, 1] == 1 / np.sqrt(2)
    # check size
    assert topoover.shape[0] == G.shape[0] and topoover.shape[1] == G.shape[-1]
    topoover_global = teneto.networkmeasures.topological_overlap(G, 'global')
    # Global = average of topoover default
    assert np.nanmean(topoover) == topoover_global
