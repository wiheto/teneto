"""

Networkmeasures: Temporal Efficiency

"""

import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path




def temporal_efficiency(data, calc='global'):
    """
    returns temporal efficiency estimate.


    **PARAMETERS**

    :data: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu'

        :paths: Dictionary of paths (output of shortest_temporal_path).


    :calc: 'global' (default) - measure averages over time and nodes.
        'node' or 'node_from' average over nodes (i) and time. Giving average efficiency for i to j.
        'node_to' measure average over nodes j and time.
         Giving average efficiency using paths to j from  i.


    **OUTPUT**

    :E: global temporal efficiency (global measure)

        :format: integer (numpy array)

    **NOTES**

    This can be implemented on a non-global level in the future.

    **SEE ALSO**

    - *shortesttemporalpath*

    **HISTORY**

    Modified - Jan 2016, WHT (documentation)
    Created - Dec 2016, WHT

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
