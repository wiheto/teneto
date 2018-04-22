"""
Network mesures: temporal degree centrality
"""

import numpy as np
import teneto.utils as utils
import warnings

def temporal_degree_centrality(net, axis=0, calc='avg', communities=None, subnet=None, decay=None):
    """

    temporal degree of network. Sum of all connections each node has through time.

    **PARAMETERS**

    :net: temporal network input (graphlet or contact).

        :nettype: 'bu', 'bd', 'wu', 'wd'

    :axis: Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    calc : str
        options: 'avg', 'time', 'module_degree_zscore'
        'avg' (returns temporal degree centrality (a 1xnode vector))
        'time' (returns a node x time matrix),
        'module_degree_zscore' returns the Z-scored within community degree centrality (communities argument required). This is done for each time-point
     i.e. 'time' returns static degree centrality per time-point.

    communities : array (Nx1)
        Vector of community assignment.
        If this is given and calc='time', then the strength within and between each communities is returned (technically not degree centrality).

    subnet : array (Nx1)
        Vector of community assignment (deprecated)

    decay : int
        if calc = 'time', then decay is possible where the centrality of
        the previous time point is carried over to the next time point but decays
        at a value of $e^decay$ such that $D_d(t+1) = e^{-decay}D_d(t) + D(t+1)$. If
        decay is 0 then the final D will equal D when calc='avg', if decay = inf
        then this will equal calc='time'.

    **OUTPUT**

    D : array
        temporal degree centrality (nodal measure). Array is 1D ('avg'), 2D ('time', 'module_degree_zscore') or 3D ('time' + communities (non-nodal/community measures))

    **SEE ALSO**

    - *temporal_closeness_centrality*

    """
    if subnet is not None:
        warnings.warn(
        "Subnet argument will be removed in v0.3.5. Use communities instead.", FutureWarning)
        communities = subnet

    # Get input in right format
    net, netinfo = utils.process_input(net, ['C', 'G', 'TO'])

    # sum sum net
    if calc == 'time' and communities is None:
        tdeg = np.squeeze(np.sum(net, axis=axis))
    elif calc != 'time' and communities is None:
        tdeg = np.sum(
            np.sum(net, axis=2), axis=axis)
    elif calc == 'module_degree_zscore' and communities is None:
        raise ValueError('Communities must be specified when calculating module degree z-score.')
    elif calc == 'module_degree_zscore' and communities is not None:
        tdeg = np.zeros([net.shape[0],net.shape[2]])
        for t in range(net.shape[2]):
            if len(communities.shape)==2:
                C = communities[:,t]
            else:
                C = communities
            for c in np.unique(C):
                k_i = np.sum(net[:, C == c,t][C== c], axis=axis)
                tdeg[C == c,t] = (k_i - np.mean(k_i)) / np.std(k_i)
        tdeg[np.isnan(tdeg)==1] = 0
    elif calc == 'time' and communities is not None:
        unique_communities = np.unique(communities)
        tdeg_communities = [np.sum(np.sum(net[communities == s1, :, :][:, communities == s2, :], axis=1), axis=0)
                       for s1 in unique_communities for s2 in unique_communities]


        tdeg = np.array(tdeg_communities)
        tdeg = np.reshape(tdeg, [len(np.unique(communities)), len(
            np.unique(communities)), net.shape[-1]])
        # Divide diagonal by 2 if undirected to correct for edges being present twice
        if netinfo['nettype'][1] == 'u':
            for s in range(tdeg.shape[0]):
                tdeg[s,s,:] = tdeg[s,s,:]/2
    else:
        raise ValueError("invalid calc argument")
    if decay and calc=='time':
        #Reshape so that time is first dimensions
        tdeg = tdeg.transpose(np.hstack([len(tdeg.shape)-1,np.arange(len(tdeg.shape)-1)]))
        for n in range(1,tdeg.shape[-1]):
            tdeg[n] = np.exp(-decay)*tdeg[n-1] + tdeg[n]
        tdeg = tdeg.transpose(np.hstack([np.arange(1,len(tdeg.shape)),0]))
    elif decay:
        print('WARNING: decay cannot be applied unless calc=time, ignoring decay')

    return tdeg
