import teneto 
import numpy as np
import pytest
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
    #Should directional also works
    C2 = teneto.networkmeasures.temporal_degree_centrality(G,axis=1)
    C3 = teneto.networkmeasures.temporal_degree_centrality(G,axis=1,calc='time')
    #With decay
    C4 = teneto.networkmeasures.temporal_degree_centrality(G,axis=1,calc='time',decay=0.5)
    #Incorrectly specified decay and warning will be raised
    C5 = teneto.networkmeasures.temporal_degree_centrality(G,axis=1,decay=0.5)
    tC4 = np.array(C3)
    for n in range(1,tC4.shape[-1]): 
        tC4[:,n] = tC4[:,n] + tC4[:,n-1] * np.exp(-0.5)
    assert (C1 == G.sum(axis=2).sum(axis=1)).all()    
    assert (C2 == G.sum(axis=2).sum(axis=0)).all()
    assert C3.shape == (3,4)
    assert (C3 == G.sum(axis=1)).all()
    assert (C3 == G.sum(axis=1)).all()
    assert (C4 == tC4).all()
    assert (C2 == C5).all()


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

def test_degreefail(): 
    G = np.zeros([3,3,4])
    G[0,1,[0,2,3]] = 1
    G[0,2,1] = 1
    G[1,2,3] = 1
    G += G.transpose([1,0,2]) 
    G = teneto.utils.set_diagonal(G,1)
    # Call different instances of temporal_degree_centrality 
    with pytest.raises(ValueError): 
        teneto.networkmeasures.temporal_degree_centrality(G,calc='module_degree_zscore')

def tdeg_with_communities(): 
    # Two graphlets templates with definite community structure 
    a=np.array([[1,1,1,0,0,0],[1,1,1,0,0,0],[1,1,1,0,0,0],[0,0,0,1,1,1],[0,0,0,1,1,1],[0,0,0,1,1,1]])
    b=np.array([[1,1,1,1,0,0],[1,1,1,1,0,0],[1,1,1,1,0,0],[1,1,1,1,0,0],[0,0,0,0,1,1],[0,0,0,0,1,1]])
    # Make into 3 time points 
    G = np.stack([a,a,b,b]).transpose([1,2,0])
    # Specify communities
    C = np.array([[0,0,0,1,1,1],[0,0,0,1,1,1],[0,0,0,0,1,1],[0,0,0,1,1,1]]).transpose()
    C1 = teneto.networkmeasures.temporal_degree_centrality(G,calc='time',communities=C)
    # SHape should be communities,communities,time 
    assert (len(np.unique(C)),len(np.unique(C)),G.shape[-1]) == C1.shape
    # Hardcode the answer which should be [[3,0],[0,3]] at t=0 and [[6,1],[1,6]] at t=2 and [[3,3],[3,1]] at t=3
    assert np.all(C1[:,:,0] == np.array([[3,0],[0,3]]))
    assert np.all(C1[:,:,2] == np.array([[6,0],[0,1]]))
    assert np.all(C1[:,:,3] == np.array([[3,3],[3,1]]))

def tdeg_with_moduledegreezscore(): 
    #module degree zscore
    a=np.array([[1,1,0,1,0,0],[1,1,1,0,0,0],[0,1,1,0,0,0],[1,0,0,1,1,1],[0,0,0,1,1,1],[0,0,0,1,1,1]])
    b=np.array([[1,1,1,1,0,0],[1,1,1,1,0,0],[1,1,1,1,0,0],[1,1,1,1,0,0],[0,0,0,0,1,1],[0,0,0,0,1,1]])
    C = np.array([[0,0,0,1,1,1],[0,0,0,1,1,1]]).transpose()
    # Make into 3 time points 
    G = np.stack([a,b]).transpose([1,2,0])
    C2 = teneto.networkmeasures.temporal_degree_centrality(G,calc='module_degree_zscore',communities=C)
    # Shape should be node x time 
    assert C2.shape == (G.shape[0],G.shape[-1])
    # Hode code correct asnwer for nodes. 
    assert C2[3,1] == (0-np.mean([1,1,0]))/np.std([1,1,0])
    assert C2[0,0] == (1-np.mean([1,2,1]))/np.std([1,2,1])
    assert C2[1,0] == (2-np.mean([1,2,1]))/np.std([1,2,1])
    assert C2[4,1] == (1-np.mean([1,1,0]))/np.std([1,1,0])
