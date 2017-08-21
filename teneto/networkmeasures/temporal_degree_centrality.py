"""
Network mesures: temporal degree centrality
"""

import numpy as np
import teneto.utils as utils


def temporal_degree_centrality(net_in, axis=0, calc='avg', subnet=None):
    """

    temporal degree of network. Sum of all connections each node has through time.

    **PARAMETERS**

    :net_in: temporal network input (graphlet or contact).

        :nettype: 'bu', 'bd', 'wu', 'wd'

    :d: Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    :calc: 'avg' (returns temporal degree centrality (a 1xnode vector))
     or 'time' (returns a node x time matrix).
     i.e. 'time' returns static degree centrality per time-point.

    :subnet: None (default) or Nx1 vector of subnetwork assignment.
    This returns a "centrality" per subnetwork instead of per node.

    **OUTPUT**

    :D: temporal degree centrality (nodal measure)

        :format: 1d numpy array (or 2d if calc = 'time')

    **SEE ALSO**

    - *temporal_closeness_centrality*

    **HISTORY**

    Modified - Mar 2017, WHT (calc='time')
    Modified - Dec 2016, WHT (calccmentation)
    Created - Nov 2016, WHT

    """

    # Get input in right format
    net_in, netinfo = utils.process_input(
        net_in, ['C', 'G', 'TO'])

    # sum sum net_in
    if calc == 'time' and not subnet:
        tdeg = np.squeeze(np.sum(net_in, axis=axis))
    elif calc != 'time' and not subnet:
        tdeg = np.sum(
            np.sum(net_in, axis=2), axis=axis)
    elif calc == 'time' and subnet:
        unique_subnet = np.unique(subnet)
        tdeg_subnet = [np.sum(np.sum(net_in[subnet == s1, :, :][:, subnet == s2, :], axis=1), axis=0)
                       for s1 in unique_subnet for s2 in unique_subnet]
        tdeg = np.array(tdeg_subnet)
        tdeg = np.reshape(tdeg, [len(np.unique(subnet)), len(
            np.unique(subnet)), net_in.shape[-1]])
    else:
        raise ValueError("invalid calc argument")

    return tdeg
