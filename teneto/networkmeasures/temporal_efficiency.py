"""

Networkmeasures: Temporal Efficiency

"""

import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path




def temporal_efficiency(data, calc='global'):
    """
    Returns temporal efficiency estimate. BU networks only.

    Parameters
    ----------
    data: dict or array
        If array, graphlet.
        If dict, either contact representation or paths (output from shortest_temporal_path).

    calc : str
        Options: 'global' (default) - measure averages over time and nodes;
        'node' or 'node_from' average over nodes (i) and time. Giving average efficiency for i to j;
        'node_to' measure average over nodes j and time;
         Giving average efficiency using paths to j from  i;

    Returns
    -------

    E : array
        Global temporal efficiency

    """

    paths = 0  # are shortest paths calculated
    if isinstance(data, dict):
        # This could be calcne better
        if [k for k in list(data.keys()) if k == 'paths'] == ['paths']:
            paths = 1
    # if shortest paths are not calculated, calculate them
    if paths == 0:
        data = shortest_temporal_path(data)

    # Calculate efficiency which is 1 over the mean path.
    if calc == 'global':
        eff = 1 / np.nanmean(data['paths'])
    elif calc == 'node' or calc == 'node_from':
        eff = 1 / np.nanmean(np.nanmean(data['paths'], axis=2), axis=1)
    elif calc == 'node_to':
        eff = 1 / np.nanmean(np.nanmean(data['paths'], axis=2), axis=0)

    return eff
