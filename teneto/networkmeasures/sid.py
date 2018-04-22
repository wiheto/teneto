

import teneto.utils as utils
import numpy as np
from teneto.networkmeasures.temporal_degree_centrality import temporal_degree_centrality


def sid(net, communities, subnet=None, axis=0, calc='global', decay=None):
    """

    Segregation integration difference (SID). An estimation of each community or global difference of within versus between community strength.

    Parameters
    ----------

    net: array, dict
        Temporal network input (graphlet or contact). Allowerd nettype: 'bu', 'bd', 'wu', 'wd'

    communities :
        a Nx1 vector or NxT array of community assignment.

    subnet : array
        a Nx1 vector or NxT array  of community assignment (will be removed in v0.3.5).

    axis : int
        Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    calc : str
        'global' returns temporal degree centrality (a 1xnode vector) (default);
         'community_pairs' returns a community x community x time matrix, which is the SID for each community pairing;
         'community_avg' (returns a community x time matrix). Which is the normalized average of each community to all other communities.

    decay: str
        if calc = 'time', then decay is possible where the centrality of
        the previous time point is carried over to the next time point but decays
        at a value of $e^decay$ such that the temporal centrality measure becomes: $D_d(t+1) = e^{-decay}D_d(t) + D(t+1)$.

    Returns
    -------

    sid: array
        segregation-integration difference. Format: 2d or 3d numpy array (depending on calc) representing (community,community,time) or (community,time)

    Source
    ------

    Fransson et al (2018) Brain network segregation and integration during an epoch-related working memory fMRI experiment.
    https://www.biorxiv.org/content/early/2018/01/23/252338

    """
    if subnet is not None:
        warnings.warn(
        "Subnet argument will be removed in v0.3.5. Use communities instead.", FutureWarning)
        communities = subnet

    net, netinfo = utils.process_input(net, ['C', 'G', 'TO'])
    D = temporal_degree_centrality(net, calc='time', communities=communities, decay=decay)
    # Check network output (order of communitiesworks)
    network_ids = np.unique(communities)
    communities_size = np.array([sum(communities==n) for n in network_ids])

    sid = np.zeros([network_ids.max()+1,network_ids.max()+1,net.shape[-1]])
    for n in network_ids:
        for m in network_ids:
            betweenmodulescaling = 1/(communities_size[n]*communities_size[m])
            if netinfo['nettype'][1] == 'd':
                withinmodulescaling = 1/(communities_size[n]*communities_size[n])
            elif netinfo['nettype'][1] == 'u':
                withinmodulescaling = 2/(communities_size[n]*(communities_size[n]-1))
                if n == m:
                    betweenmodulescaling = withinmodulescaling
            sid[n,m,:] = withinmodulescaling * D[n,n,:] - betweenmodulescaling * D[n,m,:]

    if calc == 'global':
        return np.sum(np.sum(sid,axis=1),axis=0)
    elif calc == 'communities_avg':
        return np.sum(sid,axis=axis)
    else:
        return sid
