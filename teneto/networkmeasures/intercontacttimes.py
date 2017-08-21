"""
networkmeasures: intercontacttimes
"""

import numpy as np
import teneto.utils as utils


def intercontacttimes(netin):
    """
    Calculates the intercontacttimes of each edge in a network

    **PARAMETERS**

    :netin: Temporal network (craphlet or contact).

        :nettype: 'bu', 'bd'

    **OUTPUT**

    :contacts: intercontact times as numpy array

        :format: dictionary

    **NOTES**

    Connections are assumed to be binary

    **SEE ALSO**

    *bursty_coeff*

    **History**

    :Modified: Dec 2016, WHT
    :Created: Nov 2016, WHT

    """

    # Process input
    netin, netinfo = utils.process_input(netin, ['C', 'G', 'TO'])

    if netinfo['nettype'][0] == 'd':
        print('WARNING: assuming connections to be binary when computing intercontacttimes')

    # Each time series is padded with a 0 at the start and end. Then t[0:-1]-[t:].
    # Then discard the noninformative ones (done automatically)
    # Finally return back as np array
    contacts = np.array([[None] * netinfo['netshape'][0]] * netinfo['netshape'][1])

    if netinfo['nettype'][1] == 'u':
        for i in range(0, netinfo['netshape'][0]):
            for j in range(i + 1, netinfo['netshape'][0]):
                edge_on = np.where(netin[i, j, :] > 0)[0]
                edge_on = np.append(0, edge_on)
                edge_on = np.append(edge_on, 0)
                edge_on_diff = edge_on[2:-1] - edge_on[1:-2]
                contacts[i, j] = np.array(edge_on_diff)
                contacts[j, i] = np.array(edge_on_diff)
    elif netinfo['nettype'][1] == 'd':
        for i in range(0, netinfo['netshape'][0]):
            for j in range(0, netinfo['netshape'][0]):
                edge_on = np.where(netin[i, j, :] > 0)[0]
                edge_on = np.append(0, edge_on)
                edge_on = np.append(edge_on, 0)
                edge_on_diff = edge_on[2:-1] - edge_on[1:-2]
                contacts[i, j] = np.array(edge_on_diff)

    out = {}
    out['intercontacttimes'] = contacts
    out['nettype'] = netinfo['nettype']
    return out
