import teneto
import pytest
import numpy as np
import pandas as pd


def test_errors():
    # Make sure that only 1 of three different input methods is specified
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_dict={}, from_array=np.zeros([2, 2]))
    # Make sure error raised from_array if not a numpy array
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_array=[1, 2, 3])
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_array=np.array([2]))
    # Make sure error raised from_dict if not a dictionary
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_dict=[1, 2, 3])
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_dict={})
    # Make sure error raised edge_list if not a list of lists
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_edgelist='1,2,3')
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_edgelist=[[0, 1], [0, 1, 2, 3]])
    # Make sure error raised df is not pandas
    with pytest.raises(ValueError):
        teneto.TemporalNetwork(from_df={})
    with pytest.raises(ValueError):
        df = pd.DataFrame({'i': [1, 2], 'j': [0, 1]})
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
    edgelist = [[0, 1, 2, 0.5], [0, 2, 1, 0.5]]
    tnet = teneto.TemporalNetwork(from_edgelist=edgelist)
    with pytest.raises(ValueError):
        tnet.calc_networkmeasure('skmdla')
    with pytest.raises(ValueError):
        tnet.generatenetwork('skmdla')


def test_define_tnet_unweighted():
    tnet = teneto.TemporalNetwork(nettype='wu', timetype='discrete')
    if not tnet.network.shape[1] == 4:
        raise AssertionError()
    tnet = teneto.TemporalNetwork(nettype='bu')
    if not tnet.network.shape[1] == 3:
        raise AssertionError()
    edgelist = [[0, 1, 2], [0, 2, 1]]
    tnet_edgelist = teneto.TemporalNetwork(from_edgelist=edgelist)
    if not tnet_edgelist.network.shape == (2, 3):
        raise AssertionError()
    G = np.zeros([3, 3, 3])
    G[[0, 0], [1, 2], [2, 1]] = 1
    tnet_array = teneto.TemporalNetwork(from_array=G)
    if not all(tnet_array.network == tnet_edgelist.network):
        raise AssertionError()
    tnet_df = teneto.TemporalNetwork(from_df=tnet_array.network)
    if not all(tnet_array.network == tnet_df.network):
        raise AssertionError()
    C = teneto.utils.graphlet2contact(G)
    tnet_dict = teneto.TemporalNetwork(from_dict=C)
    if not all(tnet_dict.network == tnet_edgelist.network):
        raise AssertionError()
    tnet_edgelist.add_edge([[0, 3, 1]])
    if not all(tnet_edgelist.network.iloc[-1].values == [0, 3, 1]):
        raise AssertionError()
    if not tnet_edgelist.network.shape == (3, 3):
        raise AssertionError()
    tnet_edgelist.add_edge([0, 3, 1])
    if not all(tnet_edgelist.network.iloc[-1].values == [0, 3, 1]):
        raise AssertionError()
    tnet_edgelist.drop_edge([0, 3, 1])
    if not tnet_edgelist.network.shape == (2, 3):
        raise AssertionError()


def test_define_tnet_weighted():
    tnet = teneto.TemporalNetwork(nettype='wu', timetype='discrete')
    if not tnet.network.shape[1] == 4:
        raise AssertionError()
    tnet = teneto.TemporalNetwork(nettype='bu')
    if not tnet.network.shape[1] == 3:
        raise AssertionError()
    edgelist = [[0, 1, 2, 0.5], [0, 2, 1, 0.5]]
    tnet_edgelist = teneto.TemporalNetwork(from_edgelist=edgelist)
    if not tnet_edgelist.network.shape == (2, 4):
        raise AssertionError()
    G = np.zeros([3, 3, 3])
    G[[0, 0], [1, 2], [2, 1]] = 0.5
    tnet_array = teneto.TemporalNetwork(from_array=G)
    if not all(tnet_array.network == tnet_edgelist.network):
        raise AssertionError()
    C = teneto.utils.graphlet2contact(G)
    tnet_dict = teneto.TemporalNetwork(from_dict=C)
    if not all(tnet_dict.network == tnet_edgelist.network):
        raise AssertionError()
    tnet_edgelist.add_edge([[0, 3, 1, 0.8]])
    if not all(tnet_edgelist.network.iloc[-1].values == [0, 3, 1, 0.8]):
        raise AssertionError()
    if not tnet_edgelist.network.shape == (3, 4):
        raise AssertionError()
    tnet_edgelist.drop_edge([[0, 3, 1]])
    if not tnet_edgelist.network.shape == (2, 4):
        raise AssertionError()


def test_tnet_functions():
    G = np.zeros([3, 3, 3])
    G[[0, 0], [1, 2], [2, 1]] = 1
    G = G + G.transpose([1, 0, 2])
    tnet = teneto.TemporalNetwork(from_array=G)
    G = teneto.utils.set_diagonal(G, 0)
    D = tnet.calc_networkmeasure('temporal_degree_centrality')
    if not all(G.sum(axis=-1).sum(axis=-1) == D):
        raise AssertionError()
    G = np.zeros([3, 3, 3])
    G[[0, 0], [1, 2], [2, 1]] = 0.5
    G = G + G.transpose([1, 0, 2])
    G = teneto.utils.set_diagonal(G, 0)
    tnet = teneto.TemporalNetwork(from_array=G)
    D = tnet.calc_networkmeasure('temporal_degree_centrality')
    if not all(G.sum(axis=-1).sum(axis=-1) == D):
        raise AssertionError()


def test_generatenetwork():
    tnet = teneto.TemporalNetwork()
    tnet.generatenetwork('rand_binomial', size=(5, 10), prob=1)
    if not tnet.netshape == (5, 10):
        raise AssertionError()


def test_plot():
    tnet = teneto.TemporalNetwork()
    tnet.generatenetwork('rand_binomial', size=(5, 10), prob=1)
    tnet.plot('graphlet_stack_plot')


def test_metadata():
    tnet = teneto.TemporalNetwork(nodelabels=['A', 'B', 'C'], timelabels=[
                                  0, 1, 2], desc='test meta data', starttime=0, timeunit='au')
    if not tnet.nodelabels == ['A', 'B', 'C']:
        raise AssertionError()
    if not tnet.timelabels == [0, 1, 2]:
        raise AssertionError()
    if not tnet.starttime == 0:
        raise AssertionError()
    if not tnet.desc == 'test meta data':
        raise AssertionError()
    if not tnet.timeunit == 'au':
        raise AssertionError()


def test_hdf5():
    df = pd.DataFrame({'i': [0, 0], 'j': [1, 2], 't': [0, 1]})
    tnet = teneto.TemporalNetwork(from_df=df, hdf5=True)
    if not tnet.network == './teneto_temporalnetwork.h5':
        raise AssertionError()
    df2 = pd.read_hdf('./teneto_temporalnetwork.h5')
    if not (df == df2).all().all():
        raise AssertionError()
    tnet.add_edge([0, 2, 2])
    df3 = pd.read_hdf('./teneto_temporalnetwork.h5')
    if not (df3.iloc[2].values == [0, 2, 2]).all():
        raise AssertionError()
    tnet.drop_edge([0, 2, 2])
    df4 = pd.read_hdf('./teneto_temporalnetwork.h5')
    if not (df == df4).all().all():
        raise AssertionError()


def test_hdf5_getnetwokwhen():
    df = pd.DataFrame({'i': [0, 1], 'j': [1, 2], 't': [0, 1]})
    tnet = teneto.TemporalNetwork(from_df=df, hdf5=True)
    dfcheck = tnet.get_network_when(i=0) 	
    if not (dfcheck.values == [0,1,0]).all():
        raise AssertionError()
    dfcheck = tnet.get_network_when(i=0,j=1,t=0,logic='and') 	
    if not (dfcheck.values == [0,1,0]).all():
        raise AssertionError()
    dfcheck = tnet.get_network_when(i=0,j=1,t=1,logic='or') 	
    if not (dfcheck.values == [[0, 1, 0],[1, 2, 1]]).all():
        raise AssertionError()
    dfcheck = tnet.get_network_when(t=0) 	
    if not (dfcheck.values == [0,1,0]).all():
        raise AssertionError()
    dfcheck = tnet.get_network_when(ij=1) 	
    if not (dfcheck.values == [[0, 1, 0],[1, 2, 1]]).all():
        raise AssertionError()




        elif ij is not None and t is not None and logic == 'and':
            isinstr = '(i in ' + str(ij) + ' | ' + 'j in ' + \
                str(ij) + ') & ' + 't in ' + str(t)
        elif ij is not None and t is not None and logic == 'or':
            isinstr = 'i in ' + str(ij) + ' | ' + 'j in ' + \
                str(ij) + ' | ' + 't in ' + str(t)
        elif i is not None and j is not None and logic == 'and':
            isinstr = 'i in ' + str(i) + ' & ' + 'j in ' + str(j)
        elif i is not None and t is not None and logic == 'and':
            isinstr = 'i in ' + str(i) + ' & ' + 't in ' + str(t)
        elif j is not None and t is not None and logic == 'and':
            isinstr = 'j in ' + str(j) + ' & ' + 't in ' + str(t)
        elif i is not None and j is not None and t is not None and logic == 'or':
            isinstr = 'i in ' + str(i) + ' | ' + 'j in ' + \
                str(j) + ' | ' + 't in ' + str(t)
        elif i is not None and j is not None and logic == 'or':
            isinstr = 'i in ' + str(i) + ' | ' + 'j in ' + str(j)
        elif i is not None and t is not None and logic == 'or':
            isinstr = 'i in ' + str(i) + ' | ' + 't in ' + str(t)
        elif j is not None and t is not None and logic == 'or':
            isinstr = 'j in ' + str(j) + ' | ' + 't in ' + str(t)
        elif i is not None:
            isinstr = 'i in ' + str(i)
        elif j is not None:
            isinstr = 'j in ' + str(j)
        elif t is not None:
            isinstr = 't in ' + str(t)
        elif ij is not None:
            isinstr = 'i in ' + str(ij) + ' | ' + 'j in ' + str(ij)
        df = pd.read_hdf(network, where=isinstr)
