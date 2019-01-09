"""
networkmeasures.temporal_closeness_centrality
"""

import numpy as np
from .shortest_temporal_path import shortest_temporal_path


def temporal_betweenness_centrality(tnet=None,paths=None,calc='time'):
    '''
    Returns temporal betweenness centrality per node.

    Parameters 
    -----------

    Input should be *either* tnet or paths. 

    data : array or dict 
    
        Temporal network input (graphlet or contact). nettype: 'bu', 'bd'. 

    calc : str

        either 'global' or 'time'
        
    paths : pandas dataframe
    
        Output of TenetoBIDS.networkmeasure.shortest_temporal_paths


    Returns 
    --------

    :close: array 
    
        normalized temporal betweenness centrality. 

            If calc = 'time', returns (node,time)

            If calc = 'global', returns (node)

    '''

    if tnet is not None and paths is not None: 
        raise ValueError('Only network or path input allowed.')
    if tnet is None and paths is None: 
        raise ValueError('No input.')
    # if shortest paths are not calculated, calculate them
    if tnet is not None:
        paths = shortest_temporal_path(tnet)

    bet = np.zeros([paths[['from','to']].max().max()+1,paths['t_start'].max()+1])

    for row in paths.iterrows(): 
        if (np.isnan(row[1]['path includes'])).all():
            pass
        else:
            nodes_in_path = np.unique(np.concatenate(row[1]['path includes'])).astype(int).tolist()
            nodes_in_path.remove(row[1]['from'])
            nodes_in_path.remove(row[1]['to'])
            if len(nodes_in_path)>0:
                bet[nodes_in_path,row[1]['t_start']] += 1 

    # Normalise bet
    bet = (1/((bet.shape[0]-1)*(bet.shape[0]-2))) * bet

    if calc == 'global':
        bet = np.mean(bet,axis=1)

    return bet
