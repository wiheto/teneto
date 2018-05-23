
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

def test_loadparc(): 
    teneto.utils.load_parcellation_coords('gordon2014_333')    
    teneto.utils.load_parcellation_coords('power2012_264')
    teneto.utils.load_parcellation_coords('shen2013_278')
    