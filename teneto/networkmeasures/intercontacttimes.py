import numpy as np
from teneto.utils import *


def intercontacttimes(netin):
    """
    Calculates the intercontacttimes of each edge in a network

    **PARAMETERS**

    :netin: Temporal network (craphlet or contact).

        :nettype: 'bu', 'bd'

    **OUTPUT**

    :ICT: intercontact times as numpy array

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
    netin, netinfo = process_input(netin, ['C', 'G', 'TO'])

    if netinfo['nettype'][0] == 'd':
        print('WARNING: assuming connections to be binary when computing intercontacttimes')

    # Each time series is padded with a 0 at the start and end. Then t[0:-1]-[t:]. Then discard the noninformative ones (done automatically)
    # Finally return back as np array
    ICT = np.array([[None] * netinfo['netshape'][0]] * netinfo['netshape'][1])

    if netinfo['nettype'][1] == 'u':
        for i in range(0, netinfo['netshape'][0]):
            for j in range(i + 1, netinfo['netshape'][0]):
                Aon = np.where(netin[i, j, :] > 0)[0]
                Aon = np.append(0, Aon)
                Aon = np.append(Aon, 0)
                Aon_diff = Aon[2:-1] - Aon[1:-2]
                ICT[i, j] = np.array(Aon_diff)
                ICT[j, i] = np.array(Aon_diff)
    elif netinfo['nettype'][1] == 'd':
        for i in range(0, netinfo['netshape'][0]):
            for j in range(0, netinfo['netshape'][0]):
                Aon = np.where(netin[i, j, :] > 0)[0]
                Aon = np.append(0, Aon)
                Aon = np.append(Aon, 0)
                Aon_diff = Aon[2:-1] - Aon[1:-2]
                ICT[i, j] = np.array(Aon_diff)

    out = {}
    out['intercontacttimes'] = ICT
    out['nettype'] = netinfo['nettype']
    return out
