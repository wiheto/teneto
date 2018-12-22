"""
networkmeasures.fluctuatbility
"""
import numpy as np
from ..utils import process_input


def fluctuability(netin, calc='global'):
    """
    Fluctuability of temporal networks. This is the variation of the network's edges over time.[fluct-1]_ 
    THis is the unique number of edges through time divided by the overall number of edges.

    Parameters
    ----------

    netin : array or dict 
    
        Temporal network input (graphlet or contact) (nettype: 'bd', 'bu', 'wu', 'wd')

    calc : str
        Version of fluctuabiility to calcualte. 'global'

    Returns
    -------

    fluct : array 
        Fluctuability

    Notes
    ------

    Fluctuability quantifies the variability of edges. 
    Given x number of edges, F is higher when those are repeated edges among a smaller set of edges 
    and lower when there are distributed across more edges. 

    .. math:: F = {{\sum_{i,j} H_{i,j}} \over {\sum_{i,j,t} G_{i,j,t}}}

    where H_{i,j} is a binary matrix where it is 1 if there is at least one t such that G_{i,j,t} = 1 (i.e. at least one temporal edge exists). 

    F is not normalized which makes comparisions of F across very different networks difficult (could be added). 


    Reference
    ---------

    .. [fluct-1] Thompson et al (2017) "From static to temporal network theory applications to functional brain connectivity." Network Neuroscience, 2: 1. p.69-99 [`Link <https://www.mitpressjournals.org/doi/abs/10.1162/NETN_a_00011>`_]
    
    """

    # Get input type (C or G)
    netin, netinfo = process_input(netin, ['C', 'G', 'TO'])

    netin[netin != 0] = 1
    unique_edges = np.sum(netin, axis=2)
    unique_edges[unique_edges > 0] = 1
    unique_edges[unique_edges == 0] = 0

    fluct = (np.sum(unique_edges)) / np.sum(netin)
    return fluct
