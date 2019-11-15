"""Calculates fluctuatbility"""
import numpy as np
from ..utils import process_input


def fluctuability(netin, calc='overtime'):
    r"""
    Fluctuability of temporal networks.

    This is the variation of the network's edges over time. [fluct-1]_
    This is the unique number of edges through time divided by the overall
    number of edges.

    Parameters
    ----------

    netin : array or dict

        Temporal network input (graphlet or contact)
        (nettype: 'bd', 'bu', 'wu', 'wd')

    calc : str
        Version of fluctuabiility to calcualte. 'overtime'

    Returns
    -------

    fluct : array
        Fluctuability

    Notes
    ------

    Fluctuability quantifies the variability of edges.
    Given x number of edges, F is higher when those are repeated edges among
    a smaller set of edges and lower when there are distributed across more edges.

    .. math:: F = {{\sum_{i,j} H_{i,j}} \over {\sum_{i,j,t} G_{i,j,t}}}

    where :math:`H_{i,j}` is a binary matrix where it is 1 if there is at
    least one t such that G_{i,j,t} = 1 (i.e. at least one temporal edge exists).

    F is not normalized which makes comparisions of F across very different
    networks difficult (could be added).

    Examples
    --------

    This example compares the fluctability of two different networks with the same number of edges.
    Below two temporal networks, both with 3 nodes and 3 time-points.
    Both get 3 connections.

    >>> import teneto
    >>> import numpy as np
    >>> # Manually specify node (i,j) and temporal (t) indicies.
    >>> ind_highF_i = [0,0,1]
    >>> ind_highF_j = [1,2,2]
    >>> ind_highF_t = [1,2,2]
    >>> ind_lowF_i = [0,0,0]
    >>> ind_lowF_j = [1,1,1]
    >>> ind_lowF_t = [0,1,2]
    >>> # Define 2 networks below and set above edges to 1
    >>> G_highF = np.zeros([3,3,3])
    >>> G_lowF = np.zeros([3,3,3])
    >>> G_highF[ind_highF_i,ind_highF_j,ind_highF_t] = 1
    >>> G_lowF[ind_lowF_i,ind_lowF_j,ind_lowF_t] = 1

    The two different networks look like this:

    .. plot::

        import teneto
        import numpy as np
        import matplotlib.pyplot as plt
        # Manually specify node (i,j) and temporal (t) indicies.
        ind_highF_i = [0,0,1]
        ind_highF_j = [1,2,2]
        ind_highF_t = [1,2,2]
        ind_lowF_i = [0,0,0]
        ind_lowF_j = [1,1,1]
        ind_lowF_t = [0,1,2]
        # Define 2 networks below and set above edges to 1
        G_highF = np.zeros([3,3,3])
        G_lowF = np.zeros([3,3,3])
        G_highF[ind_highF_i,ind_highF_j,ind_highF_t] = 1
        G_lowF[ind_lowF_i,ind_lowF_j,ind_lowF_t] = 1
        fig, ax = plt.subplots(1,2)
        teneto.plot.slice_plot(G_highF, ax[0], cmap='Pastel2', nodesize=20, nLabs=['0', '1', '2'])
        teneto.plot.slice_plot(G_lowF, ax[1], cmap='Pastel2', nodesize=20, nLabs=['0', '1', '2'])
        ax[0].set_title('G_highF')
        ax[1].set_title('G_lowF')
        ax[0].set_ylim([-0.25,2.25])
        ax[1].set_ylim([-0.25,2.25])
        plt.tight_layout()
        fig.show()


    Now calculate the fluctability of the two networks above.

    >>> F_high = teneto.networkmeasures.fluctuability(G_highF)
    >>> F_high
    1.0
    >>> F_low = teneto.networkmeasures.fluctuability(G_lowF)
    >>> F_low
    0.3333333333333333

    Here we see that the network with more unique connections has the higher fluctuability.

    Reference
    ---------

    .. [fluct-1]

        Thompson et al (2017)
        "From static to temporal network theory applications to
        functional brain connectivity." Network Neuroscience, 2:
        1. p.69-99
        [`Link <https://www.mitpressjournals.org/doi/abs/10.1162/NETN_a_00011>`_]

    """
    # Get input type (C or G)
    netin, _ = process_input(netin, ['C', 'G', 'TN'])

    netin[netin != 0] = 1
    unique_edges = np.sum(netin, axis=2)
    unique_edges[unique_edges > 0] = 1
    unique_edges[unique_edges == 0] = 0

    fluct = (np.sum(unique_edges)) / np.sum(netin)
    return fluct
