import teneto 
import numpy as np
def test_networkmeasures_tdc(): 
    # Make simple network
    G = np.zeros([3,3,4])
    G[0,1,[0,2,3]] = 1
    G[0,2,1] = 1
    G[1,2,3] = 1
    G += G.transpose([1,0,2]) 
    G = teneto.utils.set_diagonal(G,1)
    # Call different instances of temporal_degree_centrality 
    C1 = teneto.networkmeasures.temporal_degree_centrality(G)
    C2 = teneto.networkmeasures.temporal_degree_centrality(G,axis=1)
    C3 = teneto.networkmeasures.temporal_degree_centrality(G,axis=1,calc='time')
    C4 = teneto.networkmeasures.temporal_degree_centrality(G,axis=1,calc='time',decay=0.5)
    tC4 = np.array(C3)
    for n in range(1,tC4.shape[-1]): 
        tC4[:,n] = tC4[:,n] + tC4[:,n-1] * np.exp(-0.5)
    assert (C1 == G.sum(axis=2).sum(axis=1)).all()    
    assert (C2 == G.sum(axis=2).sum(axis=0)).all()
    assert C3.shape == (3,4)
    assert (C3 == G.sum(axis=1)).all()
    assert (C3 == G.sum(axis=1)).all()
    assert (C4 == tC4).all()


def test_sid(): 
    communities = [0,0,1,1]
    G = np.zeros([4,4,4])
    G[0,1,[0,1,3]] = 1
    G[2,3,[0,1,3]] = 1
    G[1,2,[2,3]] = 1
    G[0,3,[2,3]] = 1
    G += G.transpose([1,0,2])
    sid_g = teneto.networkmeasures.sid(G,np.array(communities))
    sid_c = teneto.networkmeasures.sid(G,np.array(communities),calc='community_pairs')
    #Calculated in head
    sid_g_true = [2,2,-1,1]    
    #Should be half sid_g since network is only calculated once
    sid_c_true = [1,1,-0.5,0.5]
    # Since only 2 networks this should be the same.
    assert np.all(sid_c[0,1,:] == sid_c_true)
    assert np.all(sid_g == sid_g_true)