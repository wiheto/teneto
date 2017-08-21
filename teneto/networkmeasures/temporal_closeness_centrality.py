"""
networkmeasures.temporal_closeness_centrality
"""

import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path


def temporal_closeness_centrality(data):
    '''
    returns temporal closeness centrality per node.
    As temporalPaths only works with binary undirected edges at the moment,
     this is required for temporal closeness centrality.

    **PARAMETERS**

    :data: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu'

        :paths: Dictionary of paths (output of shortest_temporal_path).


    **OUTPUT**

    :C: temporal closness centrality (nodal measure)

        :format: 1d numpy array

    **See Also**

    - *temporalPaths*
    - *temporalDegree*

    **History**

    Modified - Dec 2016, WHT (documentation, cleanup)
    Created - Nov 2016, WHT

    '''

    pathdata = 0  # are shortest paths calculated
    if isinstance(data, dict):
        # This could be done better
        if [k for k in list(data.keys()) if k == 'paths'] == ['paths']:
            pathdata = 1
    # if shortest paths are not calculated, calculate them
    if pathdata == 0:
        data = shortest_temporal_path(data)

    closeness = np.nansum(
        1 / np.nanmean(data['paths'], axis=2), axis=1) / (data['paths'].shape[1] - 1)

    return closeness
