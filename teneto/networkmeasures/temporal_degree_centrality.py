import numpy as np
from teneto.utils import *

def temporal_degree_centrality(netIn,d=0):
    """

    temporal degree of network. Sum of all connections each node has through time.

    **PARAMETERS**

    :netIn: temporal network input (graphlet or contact).

        :nettype: 'bu', 'bd', 'wu', 'wd'

    :d: Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    **OUTPUT**

    :D: temporal degree centrality (nodal measure)

        :format: 1d numpy array

    **SEE ALSO**

    - *temporal_closeness_centrality*

    **HISTORY**

    Modified - DEF 2016, WHT (docmentation)
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
