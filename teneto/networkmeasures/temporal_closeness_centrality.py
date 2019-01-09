"""
networkmeasures.temporal_closeness_centrality
"""

import numpy as np
from .shortest_temporal_path import shortest_temporal_path


def temporal_closeness_centrality(tnet=None,paths=None):
    '''
    Returns temporal closeness centrality per node.

    Parameters 
    -----------

    Input should be *either* tnet or paths. 

    data : array or dict 
    
        Temporal network input (graphlet or contact). nettype: 'bu', 'bd'. 
        
    paths : pandas dataframe
    
        Output of TenetoBIDS.networkmeasure.shortest_temporal_paths


    Returns 
    --------

    :close: array 
    
        temporal closness centrality (nodal measure)

    '''

    if tnet is not None and paths is not None: 
        raise ValueError('Only network or path input allowed.')
    if tnet is None and paths is None: 
        raise ValueError('No input.')
    # if shortest paths are not calculated, calculate them
    if tnet is not None:
        paths = shortest_temporal_path(tnet)

    pathmat = np.zeros([paths[['from','to']].max().max()+1, paths[['from','to']].max().max()+1, paths[['t_start']].max().max()+1]) * np.nan     
    pathmat[paths['from'].values,paths['to'].values,paths['t_start'].values] = paths['temporal-distance']

    closeness = np.nansum(1 / np.nanmean(pathmat, axis=2), axis=1) / (pathmat.shape[1] - 1)

    return closeness
