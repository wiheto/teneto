"""Calculates intercontacttimes"""

import numpy as np
from ..utils import process_input


def intercontacttimes(tnet):
    """
    Calculates the intercontacttimes of each edge in a network.

    Parameters
    -----------

    tnet : array, dict
        Temporal network (craphlet or contact). Nettype: 'bu',

    Returns
    ---------

    contacts : dict
        Intercontact times as numpy array in dictionary. contacts['intercontacttimes']

    Notes
    ------

    The inter-contact times is calculated by the time between consequecutive "active" edges (where active means
    that the value is 1 in a binary network).

    Examples
    --------

    This example goes through how inter-contact times are calculated.

    >>> import teneto
    >>> import numpy as np

    Make a network with 2 nodes and 4 time-points with 4 edges spaced out.

    >>> G = np.zeros([2,2,10])
    >>> edge_on = [1,3,5,9]
    >>> G[0,1,edge_on] = 1

    The network visualised below make it clear what the inter-contact times are between the two nodes:

    .. plot::

        import teneto
        import numpy as np
        import matplotlib.pyplot as plt
        G = np.zeros([2,2,10])
        edge_on = [1,3,5,9]
        G[0,1,edge_on] = 1
        fig, ax = plt.subplots(1, figsize=(4,2))
        teneto.plot.slice_plot(G, ax=ax, cmap='Pastel2')
        ax.set_ylim(-0.25, 1.25)
        plt.tight_layout()
        fig.show()

    Calculating the inter-contact times of these edges becomes: 2,2,4 between nodes 0 and 1.

    >>> ict = teneto.networkmeasures.intercontacttimes(G)

    The function returns a dictionary with the icts in the key: intercontacttimes. This is of the size NxN.
    So the icts between nodes 0 and 1 are found by:

    >>> ict['intercontacttimes'][0,1]
    array([2, 2, 4])

    """
    # Process input
    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')

    if tnet.nettype[0] == 'w':
        print('WARNING: assuming connections to be binary when computing intercontacttimes')

    # Each time series is padded with a 0 at the start and end.g Then t[0:-1]-[t:].
    # Then discard the noninformative ones (done automatically)
    # Finally return back as np array
    contacts = np.array([[None] * tnet.netshape[0]] * tnet.netshape[0])

    if tnet.nettype[1] == 'u':
        for i in range(0, tnet.netshape[0]):
            for j in range(i + 1, tnet.netshape[0]):
                edge_on = tnet.get_network_when(i=i, j=j)['t'].values
                if len(edge_on) > 0:
                    edge_on_diff = edge_on[1:] - edge_on[:-1]
                    contacts[i, j] = np.array(edge_on_diff)
                    contacts[j, i] = np.array(edge_on_diff)
                else:
                    contacts[i, j] = []
                    contacts[j, i] = []
    elif tnet.nettype[1] == 'd':
        for i in range(0, tnet.netshape[0]):
            for j in range(0, tnet.netshape[0]):
                edge_on = tnet.get_network_when(i=i, j=j)['t'].values
                if len(edge_on) > 0:
                    edge_on_diff = edge_on[1:] - edge_on[:-1]
                    contacts[i, j] = np.array(edge_on_diff)
                else:
                    contacts[i, j] = []

    out = {}
    out['intercontacttimes'] = contacts
    out['nettype'] = tnet.nettype
    return out
