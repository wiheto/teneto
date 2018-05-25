
import teneto
import numpy as np
import matplotlib.pyplot as plt

def test_graphletconversion():
    # For reproduceability
    np.random.seed(2018) 
    # Number of nodes
    N = 3
    # Number of timepoints
    T = 5
    # Probability of edge activation
    p0to1 = 0.2
    p1to1 = .9
    G = teneto.generatenetwork.rand_binomial([N,N,T],[p0to1, p1to1],'graphlet','bu')
    G = teneto.utils.set_diagonal(G,1)
    C = teneto.utils.graphlet2contact(G)
    G2 = teneto.utils.contact2graphlet(C)
    assert G.all() == G2.all()

def test_createtraj(): 
    traj = teneto.utils.create_traj_ranges(0,12,4)
    assert (traj == np.array([0,4,8,12],dtype=float)).all()

def test_binarize(): 
    G = np.zeros([3,3,2])
    G[0,1,0] = 0.5
    G[1,2,0] = 0.4
    G[0,2,0] = 0.3
    G[0,1,1] = 0.7    
    G[0,2,1] = 0.2    
    G[1,2,1] = 0.9
    G += G.transpose([1,0,2])
    G = teneto.utils.set_diagonal(G,1)
    Gbin_perc = teneto.utils.binarize(G,'percent',threshold_level=0.5)
    Gbin_mag = teneto.utils.binarize(G,'magnitude',threshold_level=0.45)
    G = teneto.utils.set_diagonal(G,0)
    Gt = np.zeros(G.shape) 
    Gt[G>0.45] = 1
    assert np.all(Gt == Gbin_mag)
    assert Gbin_perc[0,1,1] == Gbin_perc[1,2,1] == Gbin_perc[0,2,0] == 1 
    assert Gbin_perc[0,1,0] == Gbin_perc[1,2,0] == Gbin_perc[0,2,1] == 0 
    