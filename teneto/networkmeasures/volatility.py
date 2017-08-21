import teneto.utils as utils
import numpy as np


def volatility(net, distance_func_name='default', calc='global', subnet_id=None):
    """
    volatility of temporal networks. This is the average distance between consecutive time points of graphlets (difference is caclualted either globally, per edge)

    **PARAMETERS**

    :net: temporal network input (graphlet or contact)

        :net nettype: 'bu','bd','wu','wd'

    :D: distance function. Following options available: 'default', 'hamming', 'euclidean'. (Default implies hamming for binary networks, euclidean for weighted).
    :calc: version of volaitility to caclulate. Possibilities include:

        :'global': (default): the average distance of all nodes for each consecutive time point).
        :'edge': average distance between consecutive time points for each edge). Takes considerably longer
        :'node': (i.e. returns the average per node output when calculating volatility per 'edge').
        :'time': returns volatility per time point
        :'subnet': returns volatility per subnetwork id (see subnet_id). Also is returned per time-point and this may be changed in the future (with additional options)
        :'subnet_id': vector of integers.
        :Note: Index of subnetworks are returned "as is" with a shape of [max(subnet)+1,max(subnet)+1]. So if the indexes used are [1,2,3,5], V.shape==(6,6). The returning V[1,2] will correspond indexes 1 and 2. And missing index (e.g. here 0 and 4 will be NANs in rows and columns). If this behaviour is unwanted, call clean_subnetdexes first.

    :network:

    **OUTPUT**

    :vol: Volatility

        :format: scalar (calc='global'),
            1d numpy array (calc='node'),
            2d numpy array (calc='edge')

    **SEE ALSO**
    - *utils.getDistanceFunction*

    **HISTORY**
    Modified - Aug 2016, WHT (PEP8, cleanup)
    Modified - Dec 2016, WHT (documentation, cleanup)
    Created - Dec 2016, WHT
    """

    # Get input (C or G)
    net, netinfo = utils.process_input(net, ['C', 'G', 'TO'])

    distance_func_name = utils.check_distance_funciton_input(
        distance_func_name, netinfo)

    if not isinstance(distance_func_name, str):
        raise ValueError('Distance metric must be a string')

    # If not directional, only calc on the uppertriangle
    if netinfo['nettype'][1] == 'd':
        ind = np.triu_indices(net.shape[0], k=-net.shape[0])
    elif netinfo['nettype'][1] == 'u':
        ind = np.triu_indices(net.shape[0], k=1)

    if calc == 'subnet':
        # Make sure subnet_id is np array for indexing later on.
        subnet_id = np.array(subnet_id)
        if len(subnet_id) != netinfo['netshape'][0]:
            raise ValueError(
                'When processing per network, subnet_id vector must equal the number of nodes')
        if subnet_id.min() < 0:
            raise ValueError(
                'Subnetwork assignments must be positive integers')

    # Get chosen distance metric fucntion
    distance_func = utils.getDistanceFunction(distance_func_name)

    if calc == 'global':
        vol = np.mean([distance_func(net[ind[0], ind[1], t], net[ind[0],
                                                                       ind[1], t + 1]) for t in range(0, net.shape[-1] - 1)])
    elif calc == 'time':
        vol = [distance_func(net[ind[0], ind[1], t], net[ind[0], ind[1], t + 1])
               for t in range(0, net.shape[-1] - 1)]
    # This takes quite a bit of time to loop through. When calculating per edge/node.
    elif calc == 'edge' or calc == 'node':
        vol = np.zeros([net.shape[0], net.shape[1]])
        for i in ind[0]:
            for j in ind[1]:
                vol[i, j] = np.mean([distance_func(
                    net[i, j, t], net[i, j, t + 1]) for t in range(0, net.shape[-1] - 1)])
        if netinfo['nettype'][1] == 'u':
            vol = vol + np.transpose(vol)
        if calc == 'node':
            vol = np.sum(vol, axis=1)
    elif calc == 'subnet':
        net_id = set(subnet_id)
        vol = np.zeros([max(net_id) + 1, max(net_id) +
                        1, netinfo['netshape'][-1] - 1])
        for net1 in net_id:
            for net2 in net_id:
                vol[net1, net2, :] = [distance_func(net[subnet_id == net1][:, subnet_id == net2, t],
                                                    net[subnet_id == net1][:, subnet_id == net2, t + 1]) for t in range(0, net.shape[-1] - 1)]
    elif calc == 'withinsubnet':
        within_ind = np.array([[ind[0][n], ind[1][n]] for n in range(
            0, len(ind[0])) if subnet_id[ind[0][n]] == subnet_id[ind[1][n]]])
        vol = [distance_func(net[within_ind[:, 0], within_ind[:, 1], t], net[within_ind[:, 0],
                                                                                   within_ind[:, 1], t + 1]) for t in range(0, net.shape[-1] - 1)]
    elif calc == 'betweensubnet':
        between_ind = np.array([[ind[0][n], ind[1][n]] for n in range(
            0, len(ind[0])) if subnet_id[ind[0][n]] != subnet_id[ind[1][n]]])
        vol = [distance_func(net[between_ind[:, 0], between_ind[:, 1], t], net[between_ind[:,
                                                                                                 0], between_ind[:, 1], t + 1]) for t in range(0, net.shape[-1] - 1)]

    return vol
