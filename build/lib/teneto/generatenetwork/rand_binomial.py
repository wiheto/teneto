"""generatenetwork random binomial network"""

import numpy as np
from ..utils import graphlet2contact


def rand_binomial(size, prob, netrep='graphlet', nettype='bu', initialize='zero', netinfo=None, randomseed=None):
    """
    Creates a random binary network following a binomial distribution.

    Parameters
    ----------

    size : list or array of length 2 or 3.

        Input [n,t] generates n number of nodes and t number of time points.
        Can also be of length 3 (node x node x time) but number of nodes in 3-tuple must be identical.

    prob : int or list/array of length 2.

        If int, this indicates probabability for each node becoming active (equal for all nodes).

        If tuple/list of length 2, this indicates different probabilities for edges to become active/inactive.

            The first value is "birth rate". The probability of an absent connection becoming present.

            The second value is the "death rate". This dictates the probability of an active edge remaining present.

            example : [40,60] means there is a 40% chance that a 0 will become a 1 and a 60% chance that a 1 stays a 1.

    netrep : str
        network representation: 'graphlet' (default) or 'contact'.
    nettype : str
        Weighted or directed network. String 'bu' or 'bd' (accepts 'u' and 'd' as well as b is implicit)
    initialize : float or str
        Input percentage (in decimal) for how many nodes start activated. Alternative specify 'zero' (default) for all nodes to start deactivated.
    netinfo : dict
        Dictionary for contact representaiton information.
    randomseed : int
        Set random seed.

    Returns
    -------

    net : array or dict

        Generated nework. Format depends on netrep input argument.

    Notes
    ------

    The idea of this function is to randomly determine if an edge is present.

    Option 2 of the "prob" parameter can be used to create a small autocorrelaiton
    or make sure that, once an edge has been present, it never disapears. [rb-1]_

    Examples
    --------

    >>> import teneto
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    To make the networks a little more complex,
    the probabailities of rand_binomial can be set so differently for edges
    that have previously been active.
    Instead of passing a single integer to p, you can pass a list of 2 values.
    The first value is the probabililty for edges that,
    at t-1=0 will be active at t (is sometimes called the birth-rate).
    The second (optional) value is the probabaility of edges that,
    at t-1=1 will be active at t (sometimes called the death-rate).
    The latter value helps create an autocorrelation.
    Without it, connections will have no autocorrelation.

    **Example with just birthrate**

    Below we create a network with 5 nodes and 10 time-points.
    Edges have a 25% chance to appear.

    >>> np.random.seed(2017) # For reproduceability
    >>> N = 5 # Number of nodes
    >>> T = 10 # Number of timepoints
    >>> birth_rate = 0.25
    >>> G = teneto.generatenetwork.rand_binomial([N,N,T], [birth_rate])

    We can see that that edges appear randomly:

    >>> fig,ax = plt.subplots(figsize=(10,3))
    >>> ax = teneto.plot.slice_plot(G, ax, cmap='Set2')
    >>> fig.tight_layout()
    >>> fig.show()

    .. plot::

        import teneto
        import numpy as np
        import matplotlib.pyplot as plt
        np.random.seed(2017) # For reproduceability
        N = 5 # Number of nodes
        T = 10 # Number of timepoints
        birth_rate = 0.25
        G = teneto.generatenetwork.rand_binomial([N,N,T], [birth_rate])
        fig,ax = plt.subplots(figsize=(10,3))
        ax = teneto.plot.slice_plot(G, ax, cmap='Set2')
        fig.tight_layout()
        fig.show()

    **Example with birthrate and deathrate**

    Below we create a network with 5 nodes and 10 time-points.
    Edges have a 25% chance to appear and have a 75% chance to remain.

    >>> np.random.seed(2017) # For reproduceability
    >>> N = 5 # Number of nodes
    >>> T = 10 # Number of timepoints
    >>> birth_rate = 0.25
    >>> death_rate = 0.75
    >>> G = teneto.generatenetwork.rand_binomial([N,N,T], [birth_rate, death_rate])

    We can see the autocorrelation that this creates by plotting the network:

    >>> fig,ax = plt.subplots(figsize=(10,3))
    >>> ax = teneto.plot.slice_plot(G, ax, cmap='Set2')
    >>> fig.tight_layout()
    >>> fig.show()

    .. plot::

        import teneto
        import numpy as np
        import matplotlib.pyplot as plt
        np.random.seed(2017) # For reproduceability
        N = 5 # Number of nodes
        T = 10 # Number of timepoints
        birth_rate = 0.25
        death_rate = 0.75
        G = teneto.generatenetwork.rand_binomial([N,N,T], [birth_rate, death_rate])
        fig,ax = plt.subplots(figsize=(10,3))
        ax = teneto.plot.slice_plot(G, ax, cmap='Set2')
        fig.tight_layout()
        fig.show()

    References
    ---------

    .. [rb-1]

        Clementi et al (2008) Flooding Time in edge-Markovian Dynamic Graphs *PODC*
        This function was written without reference to this paper.
        But this paper discusses a lot of properties of these types of graphs.

    """
    size = np.atleast_1d(size)
    prob = np.atleast_1d(prob)
    if len(size) == 2 or (len(size) == 3 and size[0] == size[1]):
        pass
    else:
        raise ValueError('size input should be [numberOfNodes,Time]')
    if len(prob) > 2:
        raise ValueError('input: prob must be of len 1 or len 2')
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError('input: prob should be probability between 0 and 1')
    if nettype[-1] == 'u' or nettype[-1] == 'd':
        pass
    else:
        raise ValueError('nettype must be u or d')

    network_size = size[0]
    nr_time_points = size[-1]
    connmat = network_size * network_size
    if randomseed:
        np.random.seed(randomseed)
    if len(prob) == 1:
        net = np.random.binomial(1, prob, connmat * nr_time_points)
        net = net.reshape(network_size * network_size, nr_time_points)
    if len(prob) == 2:
        net = np.zeros([connmat, nr_time_points])
        if initialize == 'zero':
            pass
        else:
            edgesat0 = np.random.randint(
                0, connmat, int(np.round(initialize * (connmat))))
            net[edgesat0, 0] = 1
        for t_ind in range(0, nr_time_points - 1):
            edges_off = np.where(net[:, t_ind] == 0)[0]
            edges_on = np.where(net[:, t_ind] == 1)[0]
            update_edges_on = np.random.binomial(1, prob[0], len(edges_off))
            update_edge_off = np.random.binomial(1, prob[1], len(edges_on))
            net[edges_off, t_ind + 1] = update_edges_on
            net[edges_on, t_ind + 1] = update_edge_off
    # Set diagonal to 0
    net[np.arange(0, network_size * network_size, network_size + 1), :] = 0
    # Reshape to graphlet
    net = net.reshape([network_size, network_size, nr_time_points])
    # only keep upper left if nettype = u
    # Note this could be made more efficient by only doing (network_size*network_size/2-network_size) nodes
    # in connmat and inserted directly into upper triangular.
    if nettype[-1] == 'u':
        unet = np.zeros(net.shape)
        ind = np.triu_indices(network_size)
        unet[ind[0], ind[1], :] = np.array(net[ind[0], ind[1], :])
        unet = unet + np.transpose(unet, [1, 0, 2])
        net = unet
    if netrep == 'contact':
        if not netinfo:
            netinfo = {}
        netinfo['nettype'] = 'b' + nettype[-1]
        net = graphlet2contact(net, netinfo)
    return net
