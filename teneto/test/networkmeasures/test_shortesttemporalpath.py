    
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
    