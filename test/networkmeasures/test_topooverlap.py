import teneto
import numpy as np


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
    if not topoover[0, 0] == 1:
        raise AssertionError()
    # No edges persis
    if not topoover[1, 0] == 0:
        raise AssertionError()
    # When one of two edges persists
    if not topoover[0, 1] == 1 / np.sqrt(2):
        raise AssertionError()
    # check size
    if not topoover.shape[0] == G.shape[0] and topoover.shape[1] == G.shape[-1]:
        raise AssertionError()
    topoover_global = teneto.networkmeasures.topological_overlap(G, 'global')
    # Global = average of topoover default
    if not np.nanmean(topoover) == topoover_global:
        raise AssertionError()
