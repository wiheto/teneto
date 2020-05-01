import numpy as np
from ..utils import process_input, check_distance_funciton_input, get_distance_function


def volatility(tnet, distance_func='default', calc='overtime', communities=None, event_displacement=None):
    r"""
    Volatility of temporal networks.

    Volatility is the average distance between consecutive time points
    (difference is caclualted either globally or per edge).

    Parameters
    ----------

    tnet : array or dict
        temporal network input (graphlet or contact). Nettype: 'bu','bd','wu','wd'

    D : str
        Distance function. Following options available: 'default', 'hamming', 'euclidean'.
        (Default implies hamming for binary networks, euclidean for weighted).

    calc : str
        Version of volaitility to caclulate. Possibilities include:
        'overtime' - (default): the average distance of all nodes for each consecutive time point).
        'edge' - average distance between consecutive time points for each edge). Takes considerably longer
        'node' - (i.e. returns the average per node output when calculating volatility per 'edge').
        'pertime' - returns volatility per time point
        'communities' - returns volatility per communitieswork id (see communities).
        Also is returned per time-point and this may be changed in the future (with additional options)
        'event_displacement' - calculates the volatility from a specified point. Returns time-series.

    communities : array
        Array of indicies for community (eiter (node) or (node,time) dimensions).

    event_displacement : int
        if calc = event_displacement specify the temporal index where all other time-points are calculated in relation too.

    Notes
    -----

    Volatility calculates the difference between network snapshots.

    .. math:: V_t = D(G_t,G_{t+1})

    Where D is some distance function (e.g. Hamming distance for binary matrices).

    V can be calculated for the entire network (global),
    but can also be calculated for individual edges, nodes or given a community vector.

    Index of communities are returned "as is" with a shape of [max(communities)+1,max(communities)+1].
    So if the indexes used are [1,2,3,5], V.shape==(6,6).
    The returning V[1,2] will correspond indexes 1 and 2. And missing index (e.g. here 0 and 4 will be NANs in rows and columns).
    If this behaviour is unwanted, call clean_communitiesdexes first.

    Examples
    --------

    Import everything needed.

    >>> import teneto
    >>> import numpy
    >>> np.random.seed(1)
    >>> tnet = teneto.TemporalNetwork(nettype='bu')

    Here we generate a binary network where edges have a 0.5 change of going "on", and once on a 0.2 change to go "off"

    >>> tnet.generatenetwork('rand_binomial', size=(3,10), prob=(0.5,0.2))

    Calculate the volatility

    >>> tnet.calc_networkmeasure('volatility', distance_func='hamming')
    0.5555555555555556

    If we change the probabilities to instead be certain edges disapeared the time-point after the appeared:

    >>> tnet.generatenetwork('rand_binomial', size=(3,10), prob=(0.5,1))

    This will make a more volatile network

    >>> tnet.calc_networkmeasure('volatility', distance_func='hamming')
    0.1111111111111111

    We can calculate the volatility per time instead

    >>> vol_time = tnet.calc_networkmeasure('volatility', calc='pertime', distance_func='hamming')
    >>> len(vol_time)
    9
    >>> vol_time[0]
    0.3333333333333333

    Or per node:

    >>> vol_node = tnet.calc_networkmeasure('volatility', calc='node', distance_func='hamming')
    >>> vol_node
    array([0.07407407, 0.07407407, 0.07407407])

    Here we see the volatility for each node was the same.

    It is also possible to pass a community vector and the function will return volatility both within and between each community.
    So the following has two communities:

    >>> vol_com = tnet.calc_networkmeasure('volatility', calc='communities', communities=[0,1,1], distance_func='hamming')
    >>> vol_com.shape
    (2, 2, 9)
    >>> vol_com[:,:,0]
    array([[nan, 0.5],
           [0.5, 0. ]])

    And we see that, at time-point 0, there is some volatility between community 0 and 1 but no volatility within community 1.
    The reason for nan appearing is due to there only being 1 node in community 0.


    Output
    ------

    vol : array

    """
    # Get input (C or G)
    tnet, netinfo = process_input(tnet, ['C', 'G', 'TN'])

    distance_func = check_distance_funciton_input(
        distance_func, netinfo)

    if not isinstance(distance_func, str):
        raise ValueError('Distance metric must be a string')

    # If not directional, only calc on the uppertriangle
    if netinfo['nettype'][1] == 'd':
        ind = np.triu_indices(tnet.shape[0], k=-tnet.shape[0])
    elif netinfo['nettype'][1] == 'u':
        ind = np.triu_indices(tnet.shape[0], k=1)

    if calc == 'communities':
        # Make sure communities is np array for indexing later on.
        communities = np.array(communities)
        if len(communities) != netinfo['netshape'][0]:
            raise ValueError(
                'When processing per network, communities vector must equal the number of nodes')
        if communities.min() < 0:
            raise ValueError(
                'Communitiy assignments must be positive integers')

    # Get chosen distance metric fucntion
    distance_func = get_distance_function(distance_func)

    if calc == 'overtime':
        vol = np.mean([distance_func(tnet[ind[0], ind[1], t], tnet[ind[0], ind[1], t + 1])
                       for t in range(0, tnet.shape[-1] - 1)])
    elif calc == 'pertime':
        vol = [distance_func(tnet[ind[0], ind[1], t], tnet[ind[0], ind[1], t + 1])
               for t in range(0, tnet.shape[-1] - 1)]
    elif calc == 'event_displacement':
        vol = [distance_func(tnet[ind[0], ind[1], event_displacement],
                             tnet[ind[0], ind[1], t]) for t in range(0, tnet.shape[-1])]
    # This takes quite a bit of time to loop through. When calculating per edge/node.
    elif calc == 'edge' or calc == 'node':
        vol = np.zeros([tnet.shape[0], tnet.shape[1]])
        for i in ind[0]:
            for j in ind[1]:
                vol[i, j] = np.mean([distance_func(
                    tnet[i, j, t], tnet[i, j, t + 1]) for t in range(0, tnet.shape[-1] - 1)])
        if netinfo['nettype'][1] == 'u':
            vol = vol + np.transpose(vol)
        if calc == 'node':
            vol = np.mean(vol, axis=1)
    elif calc == 'communities':
        net_id = set(communities)
        vol = np.zeros([max(net_id) + 1, max(net_id) +
                        1, netinfo['netshape'][-1] - 1])
        for net1 in net_id:
            for net2 in net_id:
                if net1 != net2:
                    vol[net1, net2, :] = [distance_func(tnet[communities == net1][:, communities == net2, t].flatten(),
                                                        tnet[communities == net1][:, communities == net2, t + 1].flatten())
                                                        for t in range(0, tnet.shape[-1] - 1)]
                else:
                    nettmp = tnet[communities ==
                                  net1][:, communities == net2, :]
                    triu = np.triu_indices(nettmp.shape[0], k=1)
                    nettmp = nettmp[triu[0], triu[1], :]
                    vol[net1, net2, :] = [distance_func(nettmp[:, t].flatten(
                    ), nettmp[:, t + 1].flatten()) for t in range(0, tnet.shape[-1] - 1)]

    elif calc == 'withincommunities':
        withi = np.array([[ind[0][n], ind[1][n]] for n in range(
            0, len(ind[0])) if communities[ind[0][n]] == communities[ind[1][n]]])
        vol = [distance_func(tnet[withi[:, 0], withi[:, 1], t], tnet[withi[:, 0],
                                                                     withi[:, 1], t + 1]) for t in range(0, tnet.shape[-1] - 1)]
    elif calc == 'betweencommunities':
        beti = np.array([[ind[0][n], ind[1][n]] for n in range(
            0, len(ind[0])) if communities[ind[0][n]] != communities[ind[1][n]]])
        vol = [distance_func(tnet[beti[:, 0], beti[:, 1], t], tnet[beti[:, 0],
                                                                   beti[:, 1], t + 1]) for t in range(0, tnet.shape[-1] - 1)]

    return vol
