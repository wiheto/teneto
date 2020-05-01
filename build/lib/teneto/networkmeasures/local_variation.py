"""Networkmeasure: local_variation"""

import numpy as np
from .intercontacttimes import intercontacttimes


def local_variation(data):
    r"""
    Calculates the local variaiont of inter-contact times. [LV-1]_, [LV-2]_

    Parameters
    ----------

    data : array, dict
        This is either (1) temporal network input (graphlet or contact) with nettype: 'bu', 'bd'.
        (2) dictionary of ICTs (output of *intercontacttimes*).


    Returns
    -------
    LV : array
        Local variation per edge.


    Notes
    ------

    The local variation is like the bursty coefficient and quantifies if a series of inter-contact times are periodic, random or Poisson distributed or bursty.

    It is defined as:

    .. math:: LV = {3 \over {n-1}}\sum_{i=1}^{n-1}{{{\iota_i - \iota_{i+1}} \over {\iota_i + \iota_{i+1}}}^2}

    Where :math:`\iota` are inter-contact times and i is the index of the inter-contact time (not a node index).
    n is the number of events, making n-1 the number of inter-contact times.

    The possible range is: :math:`0 \geq LV \gt 3`.

    When periodic, LV=0, Poisson, LV=1 Larger LVs indicate bursty process.


    Examples
    ---------

    First import all necessary packages

    >>> import teneto
    >>> import numpy as np

    Now create 2 temporal network of 2 nodes and 60 time points. The first has periodict edges, repeating every other time-point:

    >>> G_periodic = np.zeros([2, 2, 60])
    >>> ts_periodic = np.arange(0, 60, 2)
    >>> G_periodic[:,:,ts_periodic] = 1

    The second has a more bursty pattern of edges:

    >>> ts_bursty = [1, 8, 9, 32, 33, 34, 39, 40, 50, 51, 52, 55]
    >>> G_bursty = np.zeros([2, 2, 60])
    >>> G_bursty[:,:,ts_bursty] = 1

    Now we call local variation for each edge.

    >>> LV_periodic = teneto.networkmeasures.local_variation(G_periodic)
    >>> LV_periodic
    array([[nan,  0.],
           [ 0., nan]])

    Above we can see that between node 0 and 1, LV=0 (the diagonal is nan).
    This is indicative of a periodic contacts (which is what we defined).
    Doing the same for the second example:

    >>> LV_bursty = teneto.networkmeasures.local_variation(G_bursty)
    >>> LV_bursty
    array([[       nan, 1.28748748],
           [1.28748748,        nan]])

    When the value is greater than 1, it indicates a bursty process.

    nans are returned if there are no intercontacttimes

    References
    ----------

    .. [LV-1]

        Shinomoto et al (2003)
        Differences in spiking patterns among cortical neurons.
        Neural Computation 15.12
        [`Link <https://www.mitpressjournals.org/doi/abs/10.1162/089976603322518759>`_]

    .. [LV-2]

        Followed eq., 4.34 in Masuda N & Lambiotte (2016)
        A guide to temporal networks. World Scientific.
        Series on Complex Networks. Vol 4
        [`Link <https://www.worldscientific.com/doi/abs/10.1142/9781786341150_0001>`_]

    """
    ict = 0  # are ict present
    if isinstance(data, dict):
        # This could be done better
        if [k for k in list(data.keys()) if k == 'intercontacttimes'] == ['intercontacttimes']:
            ict = 1
    # if shortest paths are not calculated, calculate them
    if ict == 0:
        data = intercontacttimes(data)

    if data['nettype'][1] == 'u':
        ind = np.triu_indices(data['intercontacttimes'].shape[0], k=1)
    if data['nettype'][1] == 'd':
        triu = np.triu_indices(data['intercontacttimes'].shape[0], k=1)
        tril = np.tril_indices(data['intercontacttimes'].shape[0], k=-1)
        ind = [[], []]
        ind[0] = np.concatenate([tril[0], triu[0]])
        ind[1] = np.concatenate([tril[1], triu[1]])
        ind = tuple(ind)

    ict_shape = data['intercontacttimes'].shape

    lv = np.zeros(ict_shape)

    for n in range(len(ind[0])):
        icts = data['intercontacttimes'][ind[0][n], ind[1][n]]
        # make sure there is some contact
        if len(icts) > 0:
            lv_nonnorm = np.sum(
                np.power((icts[:-1] - icts[1:]) / (icts[:-1] + icts[1:]), 2))
            lv[ind[0][n], ind[1][n]] = (3/len(icts)) * lv_nonnorm
        else:
            lv[ind[0][n], ind[1][n]] = np.nan

    # Make symetric if undirected
    if data['nettype'][1] == 'u':
        lv = lv + lv.transpose()

    for n in range(lv.shape[0]):
        lv[n, n] = np.nan

    return lv
