"""
networkmeasures.bursty_coeff
"""

import numpy as np
from teneto.networkmeasures.intercontacttimes import intercontacttimes


def bursty_coeff(data, calc='edge', nodes='all', communities=None, subnet=None):
    """
    returns calculates the bursty coefficient. Value > 0
     indicates bursty. Value < 0 periodic/tonic. Value = 0
      implies random.
    As temporalPaths only works with binary undirected edges
     at the moment, weighted edges are assumed to be binary.

    Parameters
    ----------

    data : array, dict
        This is either (1) temporal network input (graphlet or contact) with nettype: 'bu', 'bd'. (2) dictionary of ICTs (output of *intercontacttimes*).

    calc : str
        Caclulate the bursty coeff over what. Options include 'edge': calculate b_coeff on all ICTs between node i and j. (Default); 'node': caclulate b_coeff on all ICTs connected to node i.;
        'communities': calculate b_coeff for each communities (argument communities then required);
        'meanEdgePerNode': first calculate the ICTs between node i and j, then take the mean over all j.

    nodes: list or str
        Options: 'all': do for all nodes (default) or list of node indexes to calculate.

    communities : array, optoinal
        None (default) or Nx1 vector of communities assignment. This returns a "centrality" per communities instead of per node.

    subnet : array, optoinal
        None (default) or Nx1 vector of communities assignment. This returns a "centrality" per communities instead of per node. Will be removed. Use communities instead.


    Returns
    -------
    b_coeff : array
        bursty coefficienct per (edge or node measure)

    Source
    ------
    Goh and Barabasi 2008
    Discrete formulation here from Holme 2012.

    """
    if subnet is not None:
        warnings.warn(
        "Subnet argument will be removed in v0.3.5. Use communities instead.", FutureWarning)
        communities = subnet

    if calc == 'communities' and not communities:
        raise ValueError("Specified calc='communities' but no communities argument provided (list of clusters/modules)")

    ict = 0  # are ict present
    if isinstance(data, dict):
        # This could be done better
        if [k for k in list(data.keys()) if k == 'intercontacttimes'] == ['intercontacttimes']:
            ict = 1
    # if shortest paths are not calculated, calculate them
    if ict == 0:
        data = intercontacttimes(data)

    ict_shape = data['intercontacttimes'].shape

    if len(ict_shape) == 2:
        node_len = ict_shape[0] * ict_shape[1]
    elif len(ict_shape) == 1:
        node_len = 1
    else:
        raise ValueError('more than two dimensions of intercontacttimes')

    if isinstance(nodes, list) and len(ict_shape) > 1:
        node_combinations = [[list(set(nodes))[t], list(set(nodes))[tt]] for t in range(
            0, len(nodes)) for tt in range(0, len(nodes)) if t != tt]
        do_nodes = [np.ravel_multi_index(n, ict_shape) for n in node_combinations]
    else:
        do_nodes = np.arange(0, node_len)

    # Reshae ICTs
    if calc == 'node':
        ict = np.concatenate(data['intercontacttimes'][do_nodes, do_nodes], axis=1)
    elif calc == 'communities':
        unique_communities = np.unique(communities)
        ict_shape = (len(unique_communities),len(unique_communities))
        ict = np.array([[None] * ict_shape[0]] * ict_shape[1])
        for i, s1 in enumerate(unique_communities):
            for j, s2 in enumerate(unique_communities):
                if s1 == s2:
                    ind = np.triu_indices(sum(communities==s1),k=1)
                    ict[i,j] = np.concatenate(data['intercontacttimes'][ind[0],ind[1]])
                else:
                    ict[i,j] = np.concatenate(np.concatenate(data['intercontacttimes'][communities==s1,:][:,communities==s2]))
        # Quick fix, but could be better
        data['intercontacttimes'] = ict
        do_nodes = np.arange(0,ict_shape[0]*ict_shape[1])

    if len(ict_shape) > 1:
        ict = data['intercontacttimes'].reshape(ict_shape[0] * ict_shape[1])
        b_coeff = np.zeros(len(ict)) * np.nan
    else:
        b_coeff = np.zeros(1) * np.nan
        ict = [data['intercontacttimes']]

    for i in do_nodes:
        if isinstance(ict[i],np.ndarray):
            mu_ict = np.mean(ict[i])
            sigma_ict = np.std(ict[i])
            b_coeff[i] = (sigma_ict - mu_ict) / (sigma_ict + mu_ict)
        else:
            b_coeff[i] = np.nan

    if len(ict_shape) > 1:
        b_coeff = b_coeff.reshape(ict_shape)
    return b_coeff
