
import numpy as np
import teneto
import pytest


def test_randpoisson_failure():
    # Test input failures
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_poisson(3, [1, 2])
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_poisson(3, 2, [1, 2])
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_poisson(3, 2, [1, 2, 3])


def test_make_randpoisson():
    np.random.seed(2019)
    tnet = teneto.generatenetwork.rand_poisson(3, 5, 5)
    # Check there are three nodes.
    if not tnet.shape[0] == 3:
        raise AssertionError()
    # Make sure the degree centrality is 5
    ind = np.triu_indices(3, k=1)
    if not (tnet.sum(axis=-1)[ind[0], ind[1]] == [5, 5, 5]).all():
        raise AssertionError()
    C = teneto.generatenetwork.rand_poisson(
        3, 5, 5, nettype='bd', netrep='contact')
    if not C['nettype'] == 'bd':
        raise AssertionError()
