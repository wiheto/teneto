
import teneto
import numpy as np
import pytest 

def test_sp_error():
    G = np.zeros([3, 3, 4])
    G[0, 1, [0, 2, 3]] = 1
    G[0, 2, 1] = 1
    G[1, 2, 3] = 0.5
    G += G.transpose([1, 0, 2])
    G = teneto.utils.set_diagonal(G, 1)
    with pytest.raises(ValueError):
        sp = teneto.networkmeasures.shortest_temporal_path(G, quiet=0)

def test_networkmeasures_stp():
    # Make simple network
    G = np.zeros([3, 3, 4])
    G[0, 1, [0, 2, 3]] = 1
    G[0, 2, 1] = 1
    G[1, 2, 3] = 1
    G += G.transpose([1, 0, 2])
    G = teneto.utils.set_diagonal(G, 1)
    sp = teneto.networkmeasures.shortest_temporal_path(G, quiet=0)
    sp['paths'] = teneto.utils.set_diagonal(sp['paths'], 0)
    paths_true = np.zeros(sp['paths'].shape)
    # reminder dimord is from,to
    paths_true[0, 1, 0] = 1
    paths_true[0, 2, 0] = 2
    paths_true[1, 0, 0] = 1
    paths_true[1, 2, 0] = 2
    paths_true[2, 1, 0] = 3
    paths_true[2, 0, 0] = 2
    paths_true[0, 1, 1] = 2
    paths_true[0, 2, 1] = 1
    paths_true[1, 0, 1] = 2
    paths_true[1, 2, 1] = 3
    paths_true[2, 1, 1] = 2
    paths_true[2, 0, 1] = 1
    paths_true[0, 1, 2] = 1
    paths_true[0, 2, 2] = 2
    paths_true[1, 0, 2] = 1
    paths_true[1, 2, 2] = 2
    paths_true[2, 1, 2] = 2
    paths_true[2, 0, 2] = 2
    paths_true[0, 1, 3] = 1
    paths_true[0, 2, 3] = 1
    paths_true[1, 0, 3] = 1
    paths_true[1, 2, 3] = 1
    paths_true[2, 1, 3] = 1
    paths_true[2, 0, 3] = 1
    assert (sp['paths'] == paths_true).all()


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
    E2 = teneto.networkmeasures.temporal_efficiency(sp)
    assert E == E2
    # Matrix symmetric so nodal measure is same regardless of how you calculate paths
    EN1 = teneto.networkmeasures.temporal_efficiency(sp, calc='node_to')
    EN2 = teneto.networkmeasures.temporal_efficiency(sp, calc='node_from')
    assert all(EN1 == EN2)
    # Change G so matrix is directed now index 0 should be less efficient in "from" (this feature isn't implemented in teneto yet)
    #G[0,2,1] = 0
    #EN1 = teneto.networkmeasures.temporal_efficiency(G,calc='node_to')
    #EN2 = teneto.networkmeasures.temporal_efficiency(G,calc='node_from')


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
    R2 = teneto.networkmeasures.reachability_latency(sp)
    assert R == R2
    # Matrix symmetric so nodal measure is same regardless of how you calculate paths
    RN1 = teneto.networkmeasures.reachability_latency(G, calc='nodes')
    RN2 = teneto.networkmeasures.reachability_latency(sp, calc='nodes')
    assert np.all(RN1 == RN2)
    paths = teneto.utils.set_diagonal(sp['paths'], 0)
    # Rglobal is average of the longest shortest path.
    RS = paths.max(axis=0).mean(axis=1).mean()
    assert RS == R
