

import teneto.utils as utils
import numpy as np
from teneto.networkmeasures.temporal_degree_centrality import temporal_degree_centrality


def sid(net, subnet, axis=0, calc='global', decay=None):
    """

    Segregation integration difference (SID). An estimation of each subnetwork (or global) difference of within versus between subnetwork strength.

     **PARAMETERS**

     :net: temporal network input (graphlet or contact).

         :nettype: 'bu', 'bd', 'wu', 'wd'

    :axis: Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    :calc: 'global' (returns temporal degree cent rality (a 1xnode vector))
      or 'subnet_pairs' (returns a subnetwork x subnetwork x time matrix). Which is the SID for each subnet pairing
      of 'subnet_avg' (returns a subnetwork x time matrix). Which is the normalized average of each subnetwork to all other networks.


    :subnet: a Nx1 vector of subnetwork assignment.

    :decay: if calc = 'time', then decay is possible where the centrality of
    the previous time point is carried over to the next time point but decays
    at a value of $e^decay$ such that the temporal centrality measure becomes: $D_d(t+1) = e^{-decay}D_d(t) + D(t+1)$.

    **OUTPUT**

    :sid: segregation-integration difference

        :format: 2d or 3d numpy array (depending on calc) representing (subnet,subnet,time) or (subnet,time)

    **SEE ALSO**

    - *temporal_degree_centrality*

    **Source**

    Fransson et al (2018) Brain network segregation and integration during an epoch-related working memory fMRI experiment.
    https://www.biorxiv.org/content/early/2018/01/23/252338

    """

    net, netinfo = utils.process_input(net, ['C', 'G', 'TO'])
    D = temporal_degree_centrality(net, calc='time', subnet=subnet, decay=decay)
    # Check network output (order of subnetworks)
    network_ids = np.unique(subnet)
    subnet_size = np.array([sum(subnet==n) for n in network_ids])

    sid = np.zeros([network_ids.max()+1,network_ids.max()+1,net.shape[-1]])
    for n in network_ids:
        for m in network_ids:
            betweenmodulescaling = 1/(subnet_size[n]*subnet_size[m])
            if netinfo['nettype'][1] == 'd':
                withinmodulescaling = 1/(subnet_size[n]*subnet_size[n])
            elif netinfo['nettype'][1] == 'u':
                withinmodulescaling = 2/(subnet_size[n]*(subnet_size[n]-1))
                if n == m:
                    betweenmodulescaling = withinmodulescaling
            sid[n,m,:] = withinmodulescaling * D[n,n,:] - betweenmodulescaling * D[n,m,:]

    if calc == 'global':
        return np.sum(np.sum(sid,axis=1),axis=0)
    elif calc == 'subnet_avg':
        return np.sum(sid,axis=axis)
    else:
        return sid
