"""
networkmeasures.fluctuatbility
"""
import numpy as np
import teneto.utils as utils


def fluctuability(netin, calc='global'):
    """
    fluctuability of temporal networks.This is the variation in unique edges
     through time divided by the overall number of edges.

    **PARAMETERS**

    :netin: temporal network input (graphlet or contact)

        :nettype: 'bd', 'bu', 'wu', 'wd'

    :calc: version of fluctuabiility to calcualte. 'global'
     (i.e. average distance of all nodes for each consecutive time point).
      A nodal version may be added in future.

    **OUTPUT**

    :F: Fluctuability

        :format: scalar (calc='global')

    **SEE ALSO**
    - *voalitility*

    **HISTORY**

    :Modified: Jan 2016, WHT (documentation)
    :Created: Dec 2016, WHT

    """

    # Get input type (C or G)
    netin, netinfo = utils.process_input(netin, ['C', 'G', 'TO'])

    netin[netin != 0] = 1
    unique_edges = np.sum(netin, axis=2)
    unique_edges[unique_edges > 0] = 1
    unique_edges[unique_edges == 0] = 0

    fluct = (np.sum(unique_edges)) / np.sum(netin)
    return fluct
