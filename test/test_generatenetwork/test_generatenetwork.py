
import teneto
import pytest


def test_gennet_fail():
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_binomial([3, 2], [0, 1, 2])
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_binomial([3, 2, 1], [0])
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_binomial([3, 2, 1], [-1])
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_binomial([2, 2], [0], nettype='bx')
    with pytest.raises(ValueError):
        teneto.generatenetwork.rand_binomial([2, 2], [1.2], nettype='bx')


def test_gen_randbinomial():
    G = teneto.generatenetwork.rand_binomial([3, 2], [1])
    assert G.shape == (3, 3, 2)
    G = teneto.utils.set_diagonal(G, 1)
    assert G[:, :, -1].min() == 1
    G = teneto.generatenetwork.rand_binomial([3, 2], [0, 1])
    assert G.shape == (3, 3, 2)
    assert G[:, :, -1].max() == 0
    G = teneto.generatenetwork.rand_binomial([3, 1], [1], initialize=3)
    assert G[:, :, 0].max() == 1
    G = teneto.generatenetwork.rand_binomial(
        [3, 1], (1, 1), initialize=0.6666, randomseed=1)
    assert G[:, :, 0].sum() == 4
