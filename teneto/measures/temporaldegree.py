import numpy as np
from teneto.utils import *

"""

Temporal degree algorithems.

"""


def temporalDegree(netIn,d=0):
    """
    temporal degree of network.

    Parameters
    ----------
    netIn: Temporal graph of format (can be bd,bu,wu,wd):
        (i) G: graphlet (3D numpy array).
        (ii) C: contact (dictionary)
    d: Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    Returns
    ----------
    temporal degree (centrality measure)
    format: 1d numpy array

    See Also
    ----------

    History
    ----------
    Created - Nov 2016, WHT
    """

    #Get input type (C or G)
    inputType=checkInput(netIn)
    nettype = 'xx'
    #Convert C representation to G
    if inputType == 'C':
        nettype = netIn['nettype']
        netIn = contact2graphlet(netIn)
    #Get network type if not set yet
    if nettype == 'xx':
        nettype = gen_nettype(netIn)
    #Set the nodal dimension to sum over to 0, if d==1 and nettype is '.d', this gets changes to 0.
    sumOverDim = 1
    #set sumDimension to 0 if nettype is d and user specifcies d=1
    if d==1:
        sumOverDim = 0
    #sum sum netIn
    td=np.sum(np.sum(netIn,axis=2),axis=sumOverDim)
    return td
