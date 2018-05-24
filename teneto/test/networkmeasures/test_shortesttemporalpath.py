    
import teneto 
import numpy as np 
def test_networkmeasures_stp(): 
    # Make simple network
    G = np.zeros([3,3,4])
    G[0,1,[0,2,3]] = 1
    G[0,2,1] = 1
    G[1,2,3] = 1
    G += G.transpose([1,0,2]) 
    G = teneto.utils.set_diagonal(G,1)
    sp = teneto.networkmeasures.shortest_temporal_path(G)
    sp['paths'] = teneto.utils.set_diagonal(sp['paths'],0)
    paths_true = np.zeros(sp['paths'].shape)
    #reminder dimord is from,to
    paths_true[0,1,0] = 1
    paths_true[0,2,0] = 2
    paths_true[1,0,0] = 1
    paths_true[1,2,0] = 2
    paths_true[2,1,0] = 3
    paths_true[2,0,0] = 2
    paths_true[0,1,1] = 2
    paths_true[0,2,1] = 1
    paths_true[1,0,1] = 2
    paths_true[1,2,1] = 3
    paths_true[2,1,1] = 2
    paths_true[2,0,1] = 1
    paths_true[0,1,2] = 1
    paths_true[0,2,2] = 2
    paths_true[1,0,2] = 1
    paths_true[1,2,2] = 2
    paths_true[2,1,2] = 2
    paths_true[2,0,2] = 2
    paths_true[0,1,3] = 1
    paths_true[0,2,3] = 1
    paths_true[1,0,3] = 1
    paths_true[1,2,3] = 1
    paths_true[2,1,3] = 1
    paths_true[2,0,3] = 1
    assert (sp['paths'] == paths_true).all()
    
def test_networkmeasures_teff(): 
    # Test temporal efficiency  
    G = np.zeros([3,3,4])
    G[0,1,[0,2,3]] = 1
    G[0,2,1] = 1
    G[1,2,3] = 1
    G += G.transpose([1,0,2]) 
    G = teneto.utils.set_diagonal(G,1)
    E = teneto.networkmeasures.temporal_efficiency(G)
    sp = teneto.networkmeasures.shortest_temporal_path(G)
    E2 = teneto.networkmeasures.temporal_efficiency(sp)
    assert E==E2
    # Matrix symmetric so nodal measure is same regardless of how you calculate paths 
    EN1 = teneto.networkmeasures.temporal_efficiency(sp,calc='node_to')
    EN2 = teneto.networkmeasures.temporal_efficiency(sp,calc='node_from')
    assert all(EN1==EN2)
    # Change G so matrix is directed now index 0 should be less efficient in "from" (this feature isn't implemented in teneto yet)
    #G[0,2,1] = 0    
    #EN1 = teneto.networkmeasures.temporal_efficiency(G,calc='node_to')
    #EN2 = teneto.networkmeasures.temporal_efficiency(G,calc='node_from')
