"""
networkmeasures.bursty_coeff
"""

import numpy as np
from teneto.networkmeasures.intercontacttimes import intercontacttimes


def bursty_coeff(data, calc='edge', nodes='all'):
    """
    returns calculates the bursty coefficient. Value > 0
     indicates bursty. Value < 0 periodic/tonic. Value = 0
      implies random.
    As temporalPaths only works with binary undirected edges
     at the moment, weighted edges are assumed to be binary.

    **PARAMETERS**

    :data: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu', 'bd'

        :ICT: dictionary of ICTs (output of *intercontacttimes*).

    :calc: caclulate the bursty coeff over what. Options include

        :'edge': calculate b_coeff on all ICTs between node i and j. (Default)
        :'node': caclulate b_coeff on all ICTs connected to node i.
        :'meanEdgePerNode': first calculate the ICTs between node i and j,
         then take the mean over all j.

    :nodes: which do to do. Options include:

        :'all': do for all nodes (default)
        :specify: list of node indexes to calculate.


    **OUTPUT**

    :b_coeff: bursty coefficienct per (edge or node measure)

        :format: 1d numpy array

    **SEE ALSO**

    intercontacttimes

    **ORIGIN**

    Goh and b_coeffarabasi 2008
    Discrete formulation here from Holme 2012.

    **HISTORY**

    :Modified: Nov 2016, WHT (documentation)
    :Created: Nov 2016, WHT

    """

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
        do_nodes = range(0, node_len)

    # Reshae ICTs
    if calc == 'node':
        ict = np.concatenate(data['intercontacttimes']
                             [do_nodes, do_nodes], axis=1)

    if len(ict_shape) > 1:
        ict = data['intercontacttimes'].reshape(ict_shape[0] * ict_shape[1])
        b_coeff = np.zeros(len(ict)) * np.nan
    else:
        b_coeff = np.zeros(1) * np.nan
        ict = [data['intercontacttimes']]

    for i in do_nodes:
        mu_ict = np.mean(ict[i])
        sigma_ict = np.std(ict[i])
        b_coeff[i] = (sigma_ict - mu_ict) / (sigma_ict + mu_ict)
    if len(ict_shape) > 1:
        b_coeff = b_coeff.reshape(ict_shape)
    return b_coeff
