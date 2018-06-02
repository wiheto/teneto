"""
networkmeasures.fluctuatbility
"""
import numpy as np
import teneto.utils as utils


def fluctuability(netin, calc='global'):
    """
    Fluctuability of temporal networks. This is the variation of the network's edges over time. 
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

    
    """

    # Get input type (C or G)
    netin, netinfo = utils.process_input(netin, ['C', 'G', 'TO'])

    netin[netin != 0] = 1
    unique_edges = np.sum(netin, axis=2)
    unique_edges[unique_edges > 0] = 1
    unique_edges[unique_edges == 0] = 0

    fluct = (np.sum(unique_edges)) / np.sum(netin)
    return fluct
