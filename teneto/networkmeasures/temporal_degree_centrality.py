import numpy as np
from teneto.utils import *

def temporal_degree_centrality(netIn, d=0, do='avg', subnetworks=None):
    """

    temporal degree of network. Sum of all connections each node has through time.

    **PARAMETERS**

    :netIn: temporal network input (graphlet or contact).

        :nettype: 'bu', 'bd', 'wu', 'wd'

    :d: Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.

    :do: 'avg' (returns temporal degree centrality (a 1xnode vector)) or 'time' (returns a node x time matrix). i.e. 'time' returns static degree centrality per time-point

    :subnetworks: None (default) or Nx1 vector of subnetwork assignment. This returns a "centrality" per subnetwork instead of per node.  

    **OUTPUT**

    :D: temporal degree centrality (nodal measure)

        :format: 1d numpy array (or 2d if do = 'time')

    **SEE ALSO**

    - *temporal_closeness_centrality*

    **HISTORY**

    Modified - Mar 2017, WHT (do='time')
    Modified - Dec 2016, WHT (docmentation)
    Created - Nov 2016, WHT

    """

    #Get input in right format
    netIn,netInfo = process_input(netIn, ['C','G','TO'])

    #Set the nodal dimension to sum over to 0, if d==1 and nettype is '.d', this gets changes to 0.
    sumOverDim = 1
    #set sumDimension to 0 if nettype is d and user specifcies d=1
    if d==1:
        sumOverDim = 0
    #sum sum netIn
    if do == 'time' and subnetworks == None:
        tDeg = np.squeeze(np.sum(netIn, axis=sumOverDim))
    elif do != 'time' and subnetworks == None:
        tDeg = np.sum(np.sum(netIn, axis=2), axis=sumOverDim)
    elif do == 'time' and subnetworks != None:

        tDeg = np.array([np.sum(np.sum(netIn[subnetworks==sn1,:,:][:,subnetworks==sn2,:], axis=1),axis=0) for sn1 in np.unique(subnetworks)  for sn2 in np.unique(subnetworks)])
        tDeg = np.reshape(tDeg,[len(np.unique(subnetworks)), len(np.unique(subnetworks)),netIn.shape[-1]])

    return tDeg
