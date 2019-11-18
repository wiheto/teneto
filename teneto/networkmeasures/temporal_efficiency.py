"""Calculates Temporal Efficiency
"""

import numpy as np
from .shortest_temporal_path import shortest_temporal_path


def temporal_efficiency(tnet=None, paths=None, calc='overtime'):
    r"""
    Returns temporal efficiency estimate. BU networks only.

    Parameters
    ----------
    Input should be *either* tnet or paths.

    data : array or dict

        Temporal network input (graphlet or contact). nettype: 'bu', 'bd'.

    paths : pandas dataframe

        Output of TenetoBIDS.networkmeasure.shortest_temporal_paths

    calc : str
        Options: 'overtime' (default) - measure averages over time and nodes;
        'node' or 'node_from' average over nodes (i) and time. Giving average efficiency for i to j;
        'node_to' measure average over nodes j and time;
         Giving average efficiency using paths to j from  i;

    Returns
    -------

    E : array
        Global temporal efficiency

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

    # Calculate efficiency which is 1 over the mean path.
    if calc == 'overtime':
        eff = 1 / np.nanmean(pathmat)
    elif calc == 'node' or calc == 'node_from':
        eff = 1 / np.nanmean(np.nanmean(pathmat, axis=2), axis=1)
    elif calc == 'node_to':
        eff = 1 / np.nanmean(np.nanmean(pathmat, axis=2), axis=0)

    return eff
