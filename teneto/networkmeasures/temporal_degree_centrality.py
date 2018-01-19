"""
Network mesures: temporal degree centrality
"""

import numpy as np
import teneto.utils as utils


def temporal_degree_centrality(net, axis=0, calc='avg', subnet=None, decay=None):
    """

    temporal degree of network. Sum of all connections each node has through time.

    **PARAMETERS**

    :net: temporal network input (graphlet or contact).

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

    :decay: if calc = 'time', then decay is possible where the centrality of
    the previous time point is carried over to the next time point but decays
    at a value of $e^decay$ such that $D_d(t+1) = e^{-decay}D_d(t) + D(t+1)$. If
    decay is 0 then the final D will equal D when calc='avg', if decay = inf
    then this will equal calc='time'.

    **OUTPUT**

    :D: temporal degree centrality (nodal measure)

        :format: 1d numpy array (or 2d if calc = 'time')

    **SEE ALSO**

    - *temporal_closeness_centrality*

    **HISTORY**

    Modified - Mar 2017, WHT (decay added)
    Modified - Mar 2017, WHT (calc='time')
    Modified - Dec 2016, WHT (docmentation)
    Created - Nov 2016, WHT

    """

    # Get input in right format
    net, netfo = utils.process_input(
        net, ['C', 'G', 'TO'])

    # sum sum net
    if calc == 'time' and not subnet:
        tdeg = np.squeeze(np.sum(net, axis=axis))
    elif calc != 'time' and not subnet:
        tdeg = np.sum(
            np.sum(net, axis=2), axis=axis)
    elif calc == 'time' and subnet:
        unique_subnet = np.unique(subnet)
        tdeg_subnet = [np.sum(np.sum(net[subnet == s1, :, :][:, subnet == s2, :], axis=1), axis=0)
                       for s1 in unique_subnet for s2 in unique_subnet]
        tdeg = np.array(tdeg_subnet)
        tdeg = np.reshape(tdeg, [len(np.unique(subnet)), len(
            np.unique(subnet)), net.shape[-1]])
    else:
        raise ValueError("invalid calc argument")
    if decay and calc=='time':
        for n in range(1,tdeg.shape[-1]):
            tdeg[:,n] = np.exp(-decay)*tdeg[:,n-1] + tdeg[:,n]
    elif decay:
        print('WARNING: decay cannot be applied unless calc=time, ignoring decay')

    return tdeg
