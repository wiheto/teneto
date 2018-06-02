
import teneto
import numpy as np


def test_volatility():
    # Test volatility
    G = np.zeros([3, 3, 3])
    G[0, 1, [0, 1, 2]] = 1
    G[0, 2, 1] = 1
    G[1, 2, 2] = 1
    G = G + G.transpose([1, 0, 2])
    # global volatility
    v_global = teneto.networkmeasures.volatility(G)
    # v volatility per time point
    v_time = teneto.networkmeasures.volatility(G, calc='time')
    v_tr = np.array([2/6, 4/6])
    assert v_global == np.mean(v_tr)
    assert all(v_time == v_tr)
    # event displacement
    v_er_tr = np.array([0, 2/6, 2/6])
    v_er = teneto.networkmeasures.volatility(
        G, calc='event_displacement', event_displacement=0)
    assert all(v_er == v_er_tr)
    # vol per node
    v_edge = teneto.networkmeasures.volatility(G, calc='edge')
    assert np.all(v_edge == np.mean(np.abs(np.diff(G)), axis=-1))
    v_node = teneto.networkmeasures.volatility(G, calc='node')
    assert np.all(v_node == np.mean(
        np.mean(np.abs(np.diff(G)), axis=-1), axis=-1))
