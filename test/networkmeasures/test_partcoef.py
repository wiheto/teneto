import numpy as np 
import teneto 

def test_partcoef(): 

    communities = [1, 1, 1, 0]
    G = np.zeros([4, 4, 3])
    G[0, 1, [0, 2]] = 1
    G[2, 3, [0, 2]] = 1
    G[1, 2, [1, 2]] = 1
    G[0, 3, [1, 2]] = 1
    G += G.transpose([1, 0, 2])
    G[1:,1:,0] = 0.5
    part = teneto.networkmeasures.temporal_part_coeff(G, np.array(communities))
    # Hardcode the order
    assert np.all(np.argsort(part[:,0]) == [0,1,2,3])
    # Hardcode known partcoeff
    assert np.all(part[:,2] == [0.5,0,0.5,0])