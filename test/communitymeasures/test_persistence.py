import numpy as np
from teneto.communitymeasures import persistence

def test_persistence():
    temporalcommunities = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 2, 1]])
    p = persistence(temporalcommunities)
    if not np.round(p, 5) == np.round(2/3, 5):
        raise AssertionError()
    p = persistence(temporalcommunities, calc='time')
    # Ground truth
    p_out = [1, 1/3, 2/3]
    if not all(p[1:] == p_out):
        raise AssertionError()
    if not np.isnan(p[0]):
        raise AssertionError()
    p = persistence(temporalcommunities, calc='node')
    # Ground truth
    p_out = [1, 2/3, 1/3]
    if not all(p == p_out):
        raise AssertionError()
    # do multilabel communities
    temporalcommunities = np.zeros([3, 3, 3])
    temporalcommunities[0, 1, :] = [0, 1, 1]
    temporalcommunities[0, 2, :] = [1, 1, 1]
    temporalcommunities[1, 2, :] = [1, 0, 0]
    p = persistence(temporalcommunities, calc='global')
    if not np.round(p, 5) == np.round(2/3, 5):
        raise AssertionError()
    p = persistence(temporalcommunities, calc='time')
    p_out = [1/3, 1]
    if not all(p[1:] == p_out):
        raise AssertionError()
    if not np.isnan(p[0]):
        raise AssertionError()
    p = persistence(temporalcommunities, calc='node')
    # Ground truth
    p_out = [0.75, 0.5, 0.75]
    if not all(p == p_out):
        raise AssertionError()
