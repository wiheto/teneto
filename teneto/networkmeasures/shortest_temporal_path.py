"""

Networkmeasures: shortest temporal paths

"""

import numpy as np
import teneto.utils as utils


def shortest_temporal_path(netin, quiet=1):
    """
    Calculates the shortest temporal path when all possible routes can be travelled
     at each time point.
    Currently only works for binary undirected edges (but can be expanded).

    **PARAMETERS**

    :netin: temporal network input (graphlet or contact)

        :nettype: 'bu'

    :quiet: quiet (default = 1). Turn to 0 if you want progree update.

    **OUTPUT**

    :paths_dict: shortest temporal paths

        :format: dictionary

    Paths are of the struction, path['paths'][i,j,t] - shortest path for i to reach j,
     starting at time t.

    **NOTE**

    This function assumes all paths can be taken per time point.
    In a future update, this function temporalPaths will allow for
     only a portion of edges to be travelled
     per time point.
    This will be implmeneted with no change to the funcitonality of calling
     this function as it is today,
     with the defaults being all edges can be travelled.

    **SEE ALSO**

    - *temporal_efficiency*
    - *reachability_latency*
    - *temporal_closeness_centrality*

    **HISTORY**

    Modified - Aug 2017, WHT (PEP8)
    Modified - Dec 2016, WHT (documentation)
    Created - Nov 2016, WHT

    """

    # Get input type (C or G)
    # Process input
    netin, netinfo = utils.process_input(netin, ['C', 'G', 'TO'])

    if netinfo['nettype'] != 'bu':
        errormsg = ('It looks like your graph is not binary and undirected. '
                    'Shortest temporal paths can only be calculated for '
                    'binary undirected networks in Teneto at the moment. '
                    'If another type is required, please create an issue at: '
                    'github.com/wiheto/teneto.')
        raise ValueError(errormsg)

    # Preallocate output
    paths = np.zeros(netinfo['netshape']) * np.nan
    # Go backwards in time and see if something is reached
    paths_last_contact = np.zeros(
        [netinfo['netshape'][0], netinfo['netshape'][1]]) * np.nan
    for t_ind in list(reversed(range(0, netinfo['netshape'][2]))):
        if quiet == 0:
            print('--- Running for time: ' + str(t_ind) + ' ---')
        fid = np.where(netin[:, :, t_ind] >= 1)
        # Update time step
        # Note to self: Add a conditional to prevent nan warning
        # that can pop out if no path is there straight away.
        paths_last_contact += 1
        # Reset connections present to 1s
        paths_last_contact[fid[0], fid[1]] = 1
        # Update nodes with no connections
        # Nodes to update are nodes with an edge present at the time point
        for i in np.unique(fid[0]):
            connections = np.where(paths_last_contact[i, :] == 1)[0]
            #paths_last_contact_preupdate = np.array(paths_last_contact[i,:])
            paths_last_contact[i, :] = np.nanmin(
                paths_last_contact[np.hstack([connections, i]), :], axis=0)
            # make self connection nan regardless
            paths_last_contact[i, i] = np.nan
        paths[:, :, t_ind] = paths_last_contact
    # Return output
    paths_dict = {}
    percentreach = (paths.size - np.sum(np.isnan(paths))) / \
        (paths.size - (paths.shape[0] * paths.shape[2]))
    paths_dict['percentReached'] = percentreach
    paths_dict['paths'] = paths
    paths_dict['nettype'] = netinfo['nettype']

    return paths_dict
