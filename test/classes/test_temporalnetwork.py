import teneto 
import pytest 
import numpy as np 
import pandas as pd

def test_errors():
    # Make sure that only 1 of three different input methods is specified
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_dict={}, from_array=np.zeros([2,2]))
    # Make sure error raised from_array if not a numpy array
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_array=[1,2,3])
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_array=np.array([2]))
    # Make sure error raised from_dict if not a dictionary
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_dict=[1,2,3])
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_dict={})
    # Make sure error raised edge_list if not a list of lists
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_edgelist='1,2,3')
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_edgelist=[[0,1],[0,1,2,3]])
    # Make sure error raised df is not pandas
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_df={})
    with pytest.raises(ValueError):
        df = pd.DataFrame({'i':[1,2],'j':[0,1]})
        teneto.TemporalNetwork(from_df=df)
    # Make sure error raised when nettype is wrong
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(nettype='s')
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(timetype='s')
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(N='s')
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(T='s')
    edgelist = [[0,1,2,0.5],[0,2,1,0.5]]
    tnet = teneto.TemporalNetwork(from_edgelist=edgelist)
    with pytest.raises(ValueError):
        tnet.calc_networkmeasure('skmdla')
    with pytest.raises(ValueError):
        tnet.generatenetwork('skmdla')
    with pytest.raises(ValueError):
        tnet.plot('skmdla')
    with pytest.raises(ValueError):
        tnet._check_input(datain=edgelist,datatype='skmdla')


def test_define_tnet_unweighted(): 
    tnet = teneto.TemporalNetwork(nettype='wu', timetype='discrete')
    assert tnet.network.shape[1] == 4
    tnet = teneto.TemporalNetwork(nettype='bu')
    assert tnet.network.shape[1] == 3
    edgelist = [[0,1,2],[0,2,1]]
    tnet_edgelist = teneto.TemporalNetwork(from_edgelist=edgelist)
    assert tnet_edgelist.network.shape == (2,3)
    G = np.zeros([3,3,3]) 
    G[[0,0],[1,2],[2,1]] = 1
    tnet_array = teneto.TemporalNetwork(from_array=G)
    assert all(tnet_array.network == tnet_edgelist.network)
    tnet_df = teneto.TemporalNetwork(from_df=tnet_array.network)
    assert all(tnet_array.network == tnet_df.network)
    C = teneto.utils.graphlet2contact(G)
    tnet_dict = teneto.TemporalNetwork(from_dict=C)
    assert all(tnet_dict.network == tnet_edgelist.network)
    tnet_edgelist.add_edge([[0,3,1]])
    assert all(tnet_edgelist.network.iloc[-1].values == [0,3,1])
    assert tnet_edgelist.network.shape == (3,3)
    tnet_edgelist.add_edge([0,3,1])
    assert all(tnet_edgelist.network.iloc[-1].values == [0,3,1])
    tnet_edgelist.drop_edge([0,3,1])
    assert tnet_edgelist.network.shape == (2,3)
    

def test_define_tnet_weighted(): 
    tnet = teneto.TemporalNetwork(nettype='wu', timetype='discrete')
    assert tnet.network.shape[1] == 4
    tnet = teneto.TemporalNetwork(nettype='bu')
    assert tnet.network.shape[1] == 3
    edgelist = [[0,1,2,0.5],[0,2,1,0.5]]
    tnet_edgelist = teneto.TemporalNetwork(from_edgelist=edgelist)
    assert tnet_edgelist.network.shape == (2,4)
    G = np.zeros([3,3,3]) 
    G[[0,0],[1,2],[2,1]] = 0.5
    tnet_array = teneto.TemporalNetwork(from_array=G)
    assert all(tnet_array.network == tnet_edgelist.network)
    C = teneto.utils.graphlet2contact(G)
    tnet_dict = teneto.TemporalNetwork(from_dict=C)
    assert all(tnet_dict.network == tnet_edgelist.network)
    tnet_edgelist.add_edge([[0,3,1,0.8]])
    assert all(tnet_edgelist.network.iloc[-1].values == [0,3,1,0.8])
    assert tnet_edgelist.network.shape == (3,4)
    tnet_edgelist.drop_edge([[0,3,1]])
    assert tnet_edgelist.network.shape == (2,4)


def test_tnet_functions(): 
    G = np.zeros([3,3,3]) 
    G[[0,0],[1,2],[2,1]] = 1
    G = G + G.transpose([1, 0, 2])
    tnet = teneto.TemporalNetwork(from_array=G)
    G = teneto.utils.set_diagonal(G,0)
    D = tnet.calc_networkmeasure('temporal_degree_centrality')
    assert all(G.sum(axis=-1).sum(axis=-1) == D)
    G = np.zeros([3,3,3]) 
    G[[0,0],[1,2],[2,1]] = 0.5
    G = G + G.transpose([1, 0, 2])
    G = teneto.utils.set_diagonal(G,0)
    tnet = teneto.TemporalNetwork(from_array=G)
    D = tnet.calc_networkmeasure('temporal_degree_centrality')
    assert all(G.sum(axis=-1).sum(axis=-1) == D)




def test_generatenetwork():
    tnet = teneto.TemporalNetwork()
    tnet.generatenetwork('rand_binomial', size=(5,10), prob=1)
    assert tnet.netshape == (5,10)

def test_plot():
    tnet = teneto.TemporalNetwork()
    tnet.generatenetwork('rand_binomial', size=(5,10), prob=1)
    tnet.plot('graphlet_stack_plot')
