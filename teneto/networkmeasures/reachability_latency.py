"""
networkmeasures.reachability_latency
"""

import numpy as np
from .shortest_temporal_path import shortest_temporal_path


def reachability_latency(tnet=None, paths=None, rratio=1, calc='global'):
    """
    Reachability latency. This is the r-th longest temporal path.

    Parameters
    ---------

    data : array or dict

        Can either be a network (graphlet or contact), binary unidrected only.
        Alternative can be a paths dictionary (output of teneto.networkmeasure.shortest_temporal_path)

    rratio: float (default: 1)
        reachability ratio that the latency is calculated in relation to.
        Value must be over 0 and up to 1.
        1 (default) - all nodes must be reached.
        Other values (e.g. .5 imply that 50% of nodes are reached)
        This is rounded to the nearest node inter.
        E.g. if there are 6 nodes [1,2,3,4,5,6], it will be node 4 (due to round upwards)

    calc : str
        what to calculate. Alternatives: 'global' entire network; 'nodes': for each node.


    Returns
    --------
    reach_lat : array
        Reachability latency

    Notes
    ------
    Reachability latency calculates the time it takes for the paths.

    """
    if tnet is not None and paths is not None:
        raise ValueError('Only network or path input allowed.')
    if tnet is None and paths is None:
        raise ValueError('No input.')
    # if shortest paths are not calculated, calculate them
    if tnet is not None:
        paths = shortest_temporal_path(tnet)

    pathmat = np.zeros([paths[['from', 'to']].max().max(
    )+1, paths[['from', 'to']].max().max()+1, paths[['t_start']].max().max()+1]) * np.nan
    pathmat[paths['from'].values, paths['to'].values,
            paths['t_start'].values] = paths['temporal-distance']

    netshape = pathmat.shape

    edges_to_reach = netshape[0] - np.round(netshape[0] * rratio)

    reach_lat = np.zeros([netshape[1], netshape[2]]) * np.nan
    for t_ind in range(0, netshape[2]):
        paths_sort = -np.sort(-pathmat[:, :, t_ind], axis=1)
        reach_lat[:, t_ind] = paths_sort[:, edges_to_reach]
    if calc == 'global':
        reach_lat = np.nansum(reach_lat)
        reach_lat = reach_lat / ((netshape[0]) * netshape[2])
    elif calc == 'nodes':
        reach_lat = np.nansum(reach_lat, axis=1)
        reach_lat = reach_lat / (netshape[2])
    return reach_lat


def reachability_ratio(paths):
    return len(paths['temporal-distance'].dropna())/len(paths)
