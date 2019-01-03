"""
Network mesures: temporal degree centrality
"""

import numpy as np
import warnings
from ..utils import process_input, set_diagonal


def temporal_degree_centrality(tnet, axis=0, calc='avg', communities=None, decay=0, ignorediagonal=True):
    """

    temporal degree of network. Sum of all connections each node has through time.

    Parameters
    -----------

    net : array, dict
        temporal network input (graphlet or contact). Can have nettype: 'bu', 'bd', 'wu', 'wd'
    axis : int 
        Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.
    calc : str
        options: 'avg', 'time', 'module_degree_zscore'
        'avg' (returns temporal degree centrality (a 1xnode vector))
        'time' (returns a node x time matrix),
        'module_degree_zscore' returns the Z-scored within community degree centrality 
        (communities argument required). This is done for each time-point
        i.e. 'time' returns static degree centrality per time-point.
    ignorediagonal: bool
        if true, diagonal is made to 0. 
    communities : array (Nx1)
        Vector of community assignment.
        If this is given and calc='time', then the strength within and between each communities is returned (technically not degree centrality).
    decay : int
        if calc = 'time', then decay is possible where the centrality of
        the previous time point is carried over to the next time point but decays
        at a value of $e^decay$ such that $D_d(t+1) = e^{-decay}D_d(t) + D(t+1)$. If
        decay is 0 then the final D will equal D when calc='avg', if decay = inf
        then this will equal calc='time'.

    Returns
    ---------

    D : array
        temporal degree centrality (nodal measure). Array is 1D ('avg'), 2D ('time', 'module_degree_zscore') or 3D ('time' + communities (non-nodal/community measures))

    """

    # Get input in right format
    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')
    if axis == 1: 
        fromax = 'j'
        toax = 'i'
    else: 
        fromax = 'i'
        toax = 'j'
    if tnet.nettype[0] == 'b':
        tnet.network['weight'] = 1
    # Diagonal is currently deleted. 
    # if ignorediagonal:
    #     tnet = set_diagonal(tnet, 0)
    # sum sum tnet
    if calc == 'time' and communities is None:
        # Return node,time 
        tdeg = np.zeros([tnet.netshape[0], tnet.netshape[1]])
        df = tnet.network.groupby([fromax, 't']).sum().reset_index()
        tdeg[df[fromax], df['t']] = df['weight']
        # If undirected, do reverse 
        if tnet.nettype[1] == 'u':
            df = tnet.network.groupby([toax, 't']).sum().reset_index()
            tdeg[df[toax], df['t']] += df['weight']
    elif calc == 'module_degree_zscore' and communities is None:
        raise ValueError(
            'Communities must be specified when calculating module degree z-score.')
    elif calc != 'time' and communities is None:
        # Return node 
        tdeg = np.zeros([tnet.netshape[0]])
        # Strength if weighted
        df = tnet.network.groupby([fromax])['weight'].sum().reset_index()
        tdeg[df[fromax]] += df['weight']        
        # If undirected, do reverse 
        if tnet.nettype[1] == 'u':
            df = tnet.network.groupby([toax])['weight'].sum().reset_index()
            tdeg[df[toax]] += df['weight']   
    elif calc == 'module_degree_zscore' and communities is not None:
        tdeg = np.zeros([tnet.netshape[0], tnet.netshape[1]])
        for t in range(tnet.netshape[1]):
            if len(communities.shape) == 2:
                C = communities[:, t]
            else:
                C = communities
            for c in np.unique(C):
                k_i = np.sum(tnet.to_array()[:, C == c, t][C == c], axis=axis)
                tdeg[C == c, t] = (k_i - np.mean(k_i)) / np.std(k_i)
        tdeg[np.isnan(tdeg) == 1] = 0
    elif calc == 'time' and communities is not None:
        tdeg_communities = np.zeros(
            [communities.max()+1, communities.max()+1, communities.shape[-1]])
        if len(communities.shape) == 2:
            for t in range(len(communities[-1])):
                C = communities[:, t]
                unique_communities = np.unique(C)
                for s1 in unique_communities:
                    for s2 in unique_communities:
                        tdeg_communities[s1, s2, t] = np.sum(
                            np.sum(tnet.to_array()[C == s1, :, t][:, C == s2], axis=1), axis=0)
        else:
            unique_communities = np.unique(communities)
            tdeg_communities = [np.sum(np.sum(tnet.to_array()[communities == s1, :, :][:, communities == s2, :], axis=1), axis=0)
                                for s1 in unique_communities for s2 in unique_communities]

        tdeg = np.array(tdeg_communities)
        tdeg = np.reshape(tdeg, [len(np.unique(communities)), len(
            np.unique(communities)), tnet.netshape[-1]])
        # Divide diagonal by 2 if undirected to correct for edges being present twice
        if tnet.nettype[1] == 'u':
            for s in range(tdeg.shape[0]):
                tdeg[s, s, :] = tdeg[s, s, :]/2
    else:
        raise ValueError("invalid calc argument")
    if decay > 0 and calc == 'time':
        # Reshape so that time is first dimensions
        tdeg = tdeg.transpose(
            np.hstack([len(tdeg.shape)-1, np.arange(len(tdeg.shape)-1)]))
        for n in range(1, tdeg.shape[0]):
            tdeg[n] = np.exp(-decay)*tdeg[n-1] + tdeg[n]
        tdeg = tdeg.transpose(np.hstack([np.arange(1, len(tdeg.shape)), 0]))
    elif decay > 0:
        print('WARNING: decay cannot be applied unless calc=time, ignoring decay')

    return tdeg
