

import teneto.utils as utils
import numpy as np
from .temporal_degree_centrality import temporal_degree_centrality


def sid(tnet, communities, axis=0, calc='overtime', decay=0):
    r"""

    Segregation integration difference (SID). An estimation of each community or global difference of within versus between community strength.[sid-1]_

    Parameters
    ----------

    tnet: array, dict
        Temporal network input (graphlet or contact). Allowerd nettype: 'bu', 'bd', 'wu', 'wd'

    communities : array
        a Nx1 vector or NxT array of community assignment.

    axis : int
        Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    calc : str
        'overtime' returns SID over time (a 1 x community vector) (default);

        'community_pairs' returns a community x community x time matrix, which is the SID for each community pairing;

        'community_avg' (returns a community x time matrix). Which is the normalized average of each community to all other communities.

    decay: int
        if calc = 'community_pairs' or 'community_avg', then decay is possible where the centrality of
        the previous time point is carried over to the next time point but decays
        at a value of $e^decay$ such that the temporal centrality measure becomes: $D(t+1) = e^{-decay}D(t) + D(t+1)$.

    Returns
    -------

    sid: array
        segregation-integration difference. Format: 2d or 3d numpy array (depending on calc) representing (community,community,time) or (community,time)

    Notes
    ------
    SID tries to quantify if there is more segergation or intgration compared to other time-points.
    If SID > 0, then there is more segregation than usual. If SID < 0, then there is more integration than usual.

    There are three different variants of SID, one is a global measure (calc='overtime'), the second is a value per community (calc='community_avg'),
    the third is a value for each community-community pairing (calc='community_pairs').

    First we calculate the temporal strength for each edge. This is calculate by

    .. math:: S_{i,t} = \sum_j G_{i,j,t}

    The pairwise SID, when the network is undirected, is calculated by

    .. math:: SID_{A,B,t} = ({2 \over {N_A (N_A - 1)}}) S_{A,t} - ({{1} \over {N_A * N_B}}) S_{A,B,t})

    Where :math:`S_{A,t}` is the average temporal strength at time-point t for community A. :math:`N_A` is the number of nodes in community A.

    When calculating the SID for a community, it is calculated byL

    .. math:: SID_{A,t} = \sum_b^C({2 \over {N_A (N_A - 1)}}) S_{A,t} - ({{1} \over {N_A * N_b}}) S_{A,b,t})

    Where C is the number of communities.

    When calculating the SID globally, it is calculated byL

    .. math:: SID_{t} = \sum_a^C\sum_b^C({2 \over {N_a (N_a - 1)}}) S_{A,t} - ({{1} \over {N_a * N_b}}) S_{a,b,t})

    References
    -----------

    .. [sid-1]

        Fransson et al (2018) Brain network segregation and integration during an epoch-related working memory fMRI experiment.
        Neuroimage. 178. [`Link <https://www.sciencedirect.com/science/article/pii/S1053811918304476>`_]

    """
    tnet, netinfo = utils.process_input(tnet, ['C', 'G', 'TN'])
    D = temporal_degree_centrality(
        tnet, calc='pertime', communities=communities, decay=decay)

    # Check network output (order of communitiesworks)
    network_ids = np.unique(communities)
    communities_size = np.array([sum(communities == n) for n in network_ids])

    sid = np.zeros([network_ids.max()+1, network_ids.max()+1, tnet.shape[-1]])
    for n in network_ids:
        for m in network_ids:
            betweenmodulescaling = 1/(communities_size[n]*communities_size[m])
            if netinfo['nettype'][1] == 'd':
                withinmodulescaling = 1 / \
                    (communities_size[n]*communities_size[n])
            elif netinfo['nettype'][1] == 'u':
                withinmodulescaling = 2 / \
                    (communities_size[n]*(communities_size[n]-1))
                if n == m:
                    betweenmodulescaling = withinmodulescaling
            sid[n, m, :] = withinmodulescaling * \
                D[n, n, :] - betweenmodulescaling * D[n, m, :]
    # If nans emerge than there is no connection between networks at time point, so make these 0.
    sid[np.isnan(sid)] = 0

    if calc == 'overtime':
        return np.sum(np.sum(sid, axis=1), axis=0)
    elif calc == 'communities_avg':
        return np.sum(sid, axis=axis)
    else:
        return sid
