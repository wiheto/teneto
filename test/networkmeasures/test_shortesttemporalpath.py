
import teneto
import numpy as np
import pytest

# def test_sp_error():
#     G = np.zeros([3, 3, 4])
#     G[0, 1, [0, 2, 3]] = 1
#     G[0, 2, 1] = 1
#     G[1, 2, 3] = 0.5
#     G += G.transpose([1, 0, 2])
#     G = teneto.utils.set_diagonal(G, 1)
#     with pytest.raises(ValueError):
#         sp = teneto.networkmeasures.shortest_temporal_path(G, quiet=0)

# def test_networkmeasures_stp():
#     # Make simple network
#     G = np.zeros([3, 3, 4])
#     G[0, 1, [0, 2, 3]] = 1
#     G[0, 2, 1] = 1
#     G[1, 2, 3] = 1
#     G += G.transpose([1, 0, 2])
#     G = teneto.utils.set_diagonal(G, 1)
#     sp = teneto.networkmeasures.shortest_temporal_path(G, quiet=0)
#     sp['paths'] = teneto.utils.set_diagonal(sp['paths'], 0)
#     paths_true = np.zeros(sp['paths'].shape)
#     # reminder dimord is from,to
#     paths_true[0, 1, 0] = 1
#     paths_true[0, 2, 0] = 2
#     paths_true[1, 0, 0] = 1
#     paths_true[1, 2, 0] = 2
#     paths_true[2, 1, 0] = 3
#     paths_true[2, 0, 0] = 2
#     paths_true[0, 1, 1] = 2
#     paths_true[0, 2, 1] = 1
#     paths_true[1, 0, 1] = 2
#     paths_true[1, 2, 1] = 3
#     paths_true[2, 1, 1] = 2
#     paths_true[2, 0, 1] = 1
#     paths_true[0, 1, 2] = 1
#     paths_true[0, 2, 2] = 2
#     paths_true[1, 0, 2] = 1
#     paths_true[1, 2, 2] = 2
#     paths_true[2, 1, 2] = 2
#     paths_true[2, 0, 2] = 2
#     paths_true[0, 1, 3] = 1
#     paths_true[0, 2, 3] = 1
#     paths_true[1, 0, 3] = 1
#     paths_true[1, 2, 3] = 1
#     paths_true[2, 1, 3] = 1
#     paths_true[2, 0, 3] = 1
#     assert (sp['paths'] == paths_true).all()


def test_sp_new():
    G = np.zeros([5, 5, 10])
    G[0, 1, [0, 2, 3, 8]] = 1
    G[0, 2, [7, 9]] = 1
    G[1, 2, [1]] = 1
    G[1, 3, [5, 8]] = 1
    G[2, 4, [2, 4, 9]] = 1
    G[2, 3, [1, 3, 4, 7]] = 1
    # CHeck output is correct length
    paths_1step = teneto.networkmeasures.shortest_temporal_path(G, 1)
    paths_all = teneto.networkmeasures.shortest_temporal_path(G, 'all')
    dflen = np.prod(G.shape)-(G.shape[0]*G.shape[2])
    if not len(paths_all) == dflen == len(paths_1step):
        raise AssertionError()
    # Make sure these two are different
    if (paths_1step == paths_all).all()['temporal-distance']:
        raise AssertionError()
    # Check first edge and make sure all works
    paths_1step = teneto.networkmeasures.shortest_temporal_path(
        G, 1, i=0, j=3, it=0)
    paths_all = teneto.networkmeasures.shortest_temporal_path(
        G, 'all', i=0, j=3, it=0)
    paths_2step = teneto.networkmeasures.shortest_temporal_path(
        G, 2, i=0, j=3, it=0)
    if not (paths_all == paths_2step).all().all():
        raise AssertionError()
    if not paths_all['topological-distance'].values == 3:
        raise AssertionError()
    if not paths_all['temporal-distance'].values == 2:
        raise AssertionError()
    if not paths_1step['topological-distance'].values == 3:
        raise AssertionError()
    if not paths_1step['temporal-distance'].values == 4:
        raise AssertionError()
    # Check a second edge
    paths_all = teneto.networkmeasures.shortest_temporal_path(
        G, 'all', i=0, j=3, it=3)
    paths_1step = teneto.networkmeasures.shortest_temporal_path(
        G, 'all', i=0, j=3, it=3)
    if not paths_all['topological-distance'].values == 2:
        raise AssertionError()
    if not paths_all['temporal-distance'].values == 3:
        raise AssertionError()
    if not paths_1step['topological-distance'].values == 2:
        raise AssertionError()
    if not paths_1step['temporal-distance'].values == 3:
        raise AssertionError()
    # Check path is correct
    if not paths_all['path includes'].values[0] == [[0, 1], [1, 3]]:
        raise AssertionError()


def test_networkmeasures_teff():
    # Test temporal efficiency
    G = np.zeros([3, 3, 4])
    G[0, 1, [0, 2, 3]] = 1
    G[0, 2, 1] = 1
    G[1, 2, 3] = 1
    G += G.transpose([1, 0, 2])
    G = teneto.utils.set_diagonal(G, 1)
    E = teneto.networkmeasures.temporal_efficiency(G)
    sp = teneto.networkmeasures.shortest_temporal_path(G)
    E2 = teneto.networkmeasures.temporal_efficiency(paths=sp)
    if not E == E2:
        raise AssertionError()
    # Matrix symmetric so nodal measure is same regardless of how you calculate paths
    EN1 = teneto.networkmeasures.temporal_efficiency(paths=sp, calc='node_to')
    EN2 = teneto.networkmeasures.temporal_efficiency(
        paths=sp, calc='node_from')
    if not all(EN1 == EN2):
        raise AssertionError()
    # Change G so matrix is directed now index 0 should be less efficient in "from" (this feature isn't implemented in teneto yet)
    #G[0,2,1] = 0
    #EN1 = teneto.networkmeasures.temporal_efficiency(G,calc='node_to')
    #EN2 = teneto.networkmeasures.temporal_efficiency(G,calc='node_from')
    # bad inputs that will raise errors
    with pytest.raises(ValueError):
        teneto.networkmeasures.temporal_efficiency(G, paths=sp)
    with pytest.raises(ValueError):
        teneto.networkmeasures.temporal_efficiency()


def test_reachrat():
    # Test temporal efficiency
    G = np.zeros([3, 3, 4])
    G[0, 1, [0, 2, 3]] = 1
    G[0, 2, 1] = 1
    G[1, 2, 3] = 1
    G += G.transpose([1, 0, 2])
    G = teneto.utils.set_diagonal(G, 1)
    R = teneto.networkmeasures.reachability_latency(G)
    sp = teneto.networkmeasures.shortest_temporal_path(G)
    R2 = teneto.networkmeasures.reachability_latency(paths=sp)
    if not R == R2:
        raise AssertionError()
    # Matrix symmetric so nodal measure is same regardless of how you calculate paths
    RN1 = teneto.networkmeasures.reachability_latency(G, calc='nodes')
    RN2 = teneto.networkmeasures.reachability_latency(paths=sp, calc='nodes')
    if not np.all(RN1 == RN2):
        raise AssertionError()
    # bad inputs that will raise errors
    with pytest.raises(ValueError):
        teneto.networkmeasures.reachability_latency(G, paths=sp)
    with pytest.raises(ValueError):
        teneto.networkmeasures.reachability_latency()


def test_bet():
    G = np.zeros([5, 5, 3])
    G[0, 1, [0, 2, ]] = 1
    G[1, 2, [1]] = 1
    G[2, 4, [2]] = 1
    G[2, 3, [1]] = 1
    # CHeck output is correct length
    bet_time = teneto.networkmeasures.temporal_betweenness_centrality(G)
    bet_global = teneto.networkmeasures.temporal_betweenness_centrality(
        G, calc='overtime')
    if not (np.mean(bet_time, axis=1) == bet_global).all():
        raise AssertionError()
    sp = teneto.networkmeasures.shortest_temporal_path(G)
    with pytest.raises(ValueError):
        teneto.networkmeasures.reachability_latency(G, paths=sp)
    with pytest.raises(ValueError):
        teneto.networkmeasures.reachability_latency()
