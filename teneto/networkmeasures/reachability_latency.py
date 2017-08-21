"""
networkmeasures.reachability_latency
"""

import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path


def reachability_latency(data, rratio=1, calc='global'):
    """
    returns global reachability latency.
    This is the r-th longest temporal path. Where r is the number of time If r=1,

    **PARAMETERS**

    :data: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu'

        :paths: Dictionary of paths (output of shortest_temporal_path).

    :rratio: reachability ratio that the latency is calculated in relation to.
        Value must be over 0 and up to 1.
        1 (default) - all nodes must be reached.
        Other values (e.g. .5 imply that 50% of nodes are reached)
        This is rounded to the nearest node inter.
        E.g. if there are 6 nodes [1,2,3,4,5,6], it will be node 4 (due to round upwards)

    :calc: what to calculate R for:
        :'global': entire network.
        :'nodes': each node.


    **OUTPUT**

    :R: readability latency
        :format: integer (numpy array)

    **SEE ALSO**

    - *temporal_efficiency*
    - *shortest_temporal_path*

    **HISTORY**

    :Modified: Dec 2016, WHT (Documentation)
    :Created: Dec 2016, WHT

    """

    pathdata = 0  # are shortest paths calculated
    if isinstance(data, dict):
        # This could be done better
        if [k for k in list(data.keys()) if k == 'paths'] == ['paths']:
            pathdata = 1
    # if shortest paths are not calculated, calculate them
    if pathdata == 0:
        data = shortest_temporal_path(data)

    netshape = data['paths'].shape

    edges_to_reach = netshape[0] - np.round(netshape[0] * rratio)

    reach_lat = np.zeros([netshape[1], netshape[2]]) * np.nan
    for t_ind in range(0, netshape[2]):
        paths_sort = -np.sort(-data['paths'][:, :, t_ind], axis=1)
        reach_lat[:, t_ind] = paths_sort[:, edges_to_reach]
    if calc == 'global':
        reach_lat = np.nansum(reach_lat)
        reach_lat = reach_lat / ((netshape[0]) * netshape[2])
    elif calc == 'nodes':
        reach_lat = np.nansum(reach_lat, axis=1)
        reach_lat = reach_lat / (netshape[2])
    return reach_lat
