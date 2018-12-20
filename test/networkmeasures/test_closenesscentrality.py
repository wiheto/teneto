import numpy as np 
import teneto 

def test_closecoef(): 
    G = np.zeros([3, 3, 4])
    G[0, 1, [0, 2, 3]] = 1
    G[0, 2, 1] = 1
    G[1, 2, 3] = 1
    G += G.transpose([1, 0, 2])
    G = teneto.utils.set_diagonal(G, 1)
    sp = teneto.networkmeasures.shortest_temporal_path(G)
    close1 = teneto.networkmeasures.temporal_closeness_centrality(G)
    close2 = teneto.networkmeasures.temporal_closeness_centrality(sp)
    assert np.all(close1 == close2)
    assert np.all(np.nansum(1/np.nanmean(sp['paths'],axis=2),axis=1)*(1/(3-1)) == close1)
