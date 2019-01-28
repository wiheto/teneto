import community 
import pandas as pd 
import numpy as np 
from ..utils import process_input
from ..utils import tnet_to_nx
from ..utils import create_supraadjacency_matrix



def temporal_louvain(tnet, resolution=1, intersliceweight=1):
    """
    Louvain clustering for a temporal network

    Parameters 
    -----------
    tnet : array, dict, TemporalNetwork
        Input network
    resolution : int 
        resolution of Louvain clustering ($\gamma$)
    interslice : int
        interslice weight of multilayer clustering ($\omega$)

    Returns 
    -------
    communities : array (node,time)
        node,time array of community assignment

    Note 
    ---- 
    Negative edges are not currently dealt with and has to be done outside of function. 
    """

    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')
    supranet = create_supraadjacency_matrix(tnet, intersliceweight=intersliceweight)
    nxsupra = tnet_to_nx(supranet)
    com = community.best_partition(nxsupra, resolution=resolution)
    communities = np.reshape(list(com.values()),[tnet.N,tnet.T])
    return communities