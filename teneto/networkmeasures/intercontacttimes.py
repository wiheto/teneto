"""
networkmeasures: intercontacttimes
"""

import numpy as np
from ..utils import process_input


def intercontacttimes(tnet):
    """
    Calculates the intercontacttimes of each edge in a network. 

    Parameters
    -----------

    tnet : array, dict
        Temporal network (craphlet or contact). Nettype: 'bu', 'bd'

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

    The network looks like this: 

    .. plot:: 
        import teneto 
        import numpy as np     
        G = np.zeros([2,2,10])
        edge_on = [1,3,5,9]
        G[0,1,edge_on] = 1
        fig, ax = plt.subplots(1)
        teneto.plot.slice_plot(G, ax=ax)
        fig.show()

    Calculating the inter-contact times of these edges becomes: 2,2,4 between nodes 0 and 1. 

    >>> ict = teneto.networkmeasures.intercontacttimes(G)

    The function returns a dictionary with the icts in the key: intercontacttimes. This is of the size NxN. 
    So the icts between nodes 0 and 1 are found by:   

    >>> ict['intercontacttimes'][0,1]
    array([2, 2, 4])

    """

    # Process input
    tnet, netinfo = process_input(tnet, ['C', 'G', 'TO'])

    if netinfo['nettype'][0] == 'd':
        print('WARNING: assuming connections to be binary when computing intercontacttimes')

    # Each time series is padded with a 0 at the start and end. Then t[0:-1]-[t:].
    # Then discard the noninformative ones (done automatically)
    # Finally return back as np array
    contacts = np.array([[None] * netinfo['netshape'][0]] * netinfo['netshape'][1])

    if netinfo['nettype'][1] == 'u':
        for i in range(0, netinfo['netshape'][0]):
            for j in range(i + 1, netinfo['netshape'][0]):
                edge_on = np.where(tnet[i, j, :] > 0)[0]
                edge_on = np.append(0, edge_on)
                edge_on = np.append(edge_on, 0)
                edge_on_diff = edge_on[2:-1] - edge_on[1:-2]
                contacts[i, j] = np.array(edge_on_diff)
                contacts[j, i] = np.array(edge_on_diff)
    elif netinfo['nettype'][1] == 'd':
        for i in range(0, netinfo['netshape'][0]):
            for j in range(0, netinfo['netshape'][0]):
                edge_on = np.where(tnet[i, j, :] > 0)[0]
                edge_on = np.append(0, edge_on)
                edge_on = np.append(edge_on, 0)
                edge_on_diff = edge_on[2:-1] - edge_on[1:-2]
                contacts[i, j] = np.array(edge_on_diff)

    out = {}
    out['intercontacttimes'] = contacts
    out['nettype'] = netinfo['nettype']
    return out
