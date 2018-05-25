
import teneto 
import numpy as np
import pytest
def test_bursty():
    t1 = np.arange(0,60,2)
    t2 = [1,8,9,32,33,34,39,40,50,51,52,55] 
    
    G = np.zeros([3,3,60])
    G[0,1,t1] = 1
    G[1,2,t2] = 1
    ict = teneto.networkmeasures.intercontacttimes(G)
    G += G.transpose([1,0,2]) 
    ict2 = teneto.networkmeasures.intercontacttimes(G)
    assert all(np.diff(t1) == ict['intercontacttimes'][0,1])
    assert all(np.diff(t1) == ict2['intercontacttimes'][0,1])
    assert all(np.diff(t2) == ict['intercontacttimes'][1,2])
    assert all(np.diff(t2) == ict2['intercontacttimes'][1,2])
    B1 = teneto.networkmeasures.bursty_coeff(ict)
    B2 = teneto.networkmeasures.bursty_coeff(G)
    B3 = teneto.networkmeasures.bursty_coeff(ict,nodes=[0,1])
    assert B1[0,1] == B2[0,1]    
    assert B1[1,2] == B2[1,2]
    assert B1[0,1] == -1 
    assert B1[0,1] == B3[0,1]
    assert np.isnan(B3[1,2]) == 1
    assert B1[1,2] == (np.std(np.diff(t2))-np.mean(np.diff(t2)))/(np.mean(np.diff(t2))+np.std(np.diff(t2)))
        
    with pytest.raises(ValueError): 
        teneto.networkmeasures.bursty_coeff(G,calc='communities')