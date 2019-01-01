
import teneto
import numpy as np
import matplotlib.pyplot as plt
import pytest


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
    Gbin_perc2 = teneto.utils.binarize(G,'percent',threshold_level=0.5,axis='graphlet')
    Gbin_mag = teneto.utils.binarize(G,'magnitude',threshold_level=0.45)
    Gbin_mag_c = teneto.utils.binarize(teneto.utils.graphlet2contact(G),'magnitude',threshold_level=0.45)
    Gbin_perc_c = teneto.utils.binarize(teneto.utils.graphlet2contact(G),'percent',threshold_level=0.5)
    Gbin_mag = teneto.utils.binarize(G,'magnitude',threshold_level=0.45)
    G = teneto.utils.set_diagonal(G,0)
    Gt = np.zeros(G.shape) 
    Gt[G>0.45] = 1
    assert np.all(Gt == Gbin_mag)
    assert Gbin_perc[0,1,1] == Gbin_perc[1,2,1] == Gbin_perc[0,2,0] == 1 
    assert Gbin_perc[0,1,0] == Gbin_perc[1,2,0] == Gbin_perc[0,2,1] == 0 
    assert Gbin_perc2[0,1,0] == Gbin_perc2[1,2,1] == 1 
    assert Gbin_perc2[0,1,1] == Gbin_perc2[1,2,0] == Gbin_perc2[0,2,1] == Gbin_perc2[0,1,1] == 0 
    assert isinstance(Gbin_mag_c,dict)
    assert isinstance(Gbin_perc_c,dict)
    assert np.all(teneto.utils.contact2graphlet(Gbin_perc_c) == Gbin_perc)
    assert np.all(teneto.utils.contact2graphlet(Gbin_mag_c) == Gbin_mag)

def test_binarize_rdp(): 
    G = np.zeros([2,2,5])
    G[0,1,0] = 1
    G[0,1,1] = 0
    G[0,1,2] = 0.5
    G[0,1,3] = 0    
    G[0,1,4] = 1    
    Gbin1 = teneto.utils.binarize(G,'rdp',0.49)
    Gbin2 = teneto.utils.binarize(G,'rdp',0.51)
    assert (np.all(Gbin1[0,1,:] == [1,0,1,0,1]))
    assert (np.all(Gbin2[0,1,:] == [1,0,0,0,1]))



def test_contactmultiplevalues():

    g = teneto.generatenetwork.rand_binomial([2,1],1,initialize=1) 
    c = teneto.utils.graphlet2contact(g)
    c['contacts']=np.vstack([c['contacts'],c['contacts']])
    cnew = teneto.utils.multiple_contacts_get_values(c)
    assert cnew['values'] == [2]    
    assert len(cnew['contacts']) == 1


def test_getdimord(): 
    dimord = teneto.utils.get_dimord('volatility','global')
    assert dimord == 'time'
    # Need to impove a lot of the dimord settings!!


def test_cleancommunityindicies(): 
    c = [5,3,4,5,5,5,3,3,3,4,4]
    c_expected = [2,0,1,2,2,2,0,0,0,1,1]
    c_cleaned = teneto.utils.clean_community_indexes(c)
    assert np.all(c_cleaned == c_expected)


def test_load_parc_cords(): 
    parc = teneto.utils.load_parcellation_coords('power2012_264')
    assert parc.shape == (264,3)


def test_contact2graphletfail():

    C = {} 
    # Error that dimord must be present
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['dimord'] = 'blablabla'
    # Error that dimord must be correct
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['dimord'] = 'node,node,time'
    # Error that nettype is missing
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['nettype'] = 'blablablabl'
    #Specify incorrect nettype
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['nettype'] = 'wu'
    #Netshape missing
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['netshape'] = 'wu'
    #Netshape incorrectly specified
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['netshape'] = (3,10)
    #Netshape not long enough
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['netshape'] = (3,3,10)
    #values field is missing, since weighted
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['values'] = [1]
    #contacts field missing
    with pytest.raises(ValueError):
        teneto.utils.contact2graphlet(C)
    C['contacts'] = [[0,1,1]]
    C['timetype'] = 'discrete'
    C['diagonal'] = 1
    teneto.utils.contact2graphlet(C)




def test_graphlet2contactfail():

    G = np.zeros([3,2])
    # 0th and 1st Dimensions should be equal
    with pytest.raises(ValueError):
        teneto.utils.graphlet2contact(G)
    G = np.zeros([2,2,3,2])
    # Cannot be more than 3 dimensions
    with pytest.raises(ValueError):
        teneto.utils.graphlet2contact(G)
    G = np.zeros([2,2])
    params = {'nLabs': ['a']} 
    with pytest.raises(ValueError):
        teneto.utils.graphlet2contact(G,params)   
    G = np.zeros([2,2])
    params = {'t0': [1,2]} 
    with pytest.raises(ValueError):
        teneto.utils.graphlet2contact(G,params)       
    G = np.zeros([2,2])
    params = {'nettype': 'mynetwork'} 
    with pytest.raises(ValueError):
        teneto.utils.graphlet2contact(G,params)   


def test_utils_fails(): 
    # When unknown distance function is specified
    with pytest.raises(ValueError):
        teneto.utils.getDistanceFunction('blabla')   
