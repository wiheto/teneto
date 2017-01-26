from teneto.utils import *
import numpy as np

def volatility(netIn,D='default',do='global'):
    """
    volatility of temporal networks. This is the average distance between consecutive time points of graphlets (difference is caclualted either globally, per edge)

    **PARAMETERS**

    :netIn: temporal network input (graphlet or contact)

        :netIn nettype: 'bu','bd','wu','wd'

    :D: distance function. Following options available: 'default', 'hamming', 'euclidean'. (Default implies hamming for binary networks, euclidean for weighted).
    :do: version of volaitility to caclulate. Possibilities include:

        :'global': (default): the average distance of all nodes for each consecutive time point).
        :'edge': average distance between consecutive time points for each edge). Takes considerably longer
        :'node': (i.e. returns the average per node output when calculating volatility per 'edge').
        : 'time': returns volatility per time point

    **OUTPUT**

    :V: Volatility

        :format: scalar (do='global'),
            1d numpy array (do='node'),
            2d numpy array (do='edge')

    **SEE ALSO**
    - *utils.getDistanceFunction*

    **HISTORY**
    Modified - Dec 2016, WHT (documentation, cleanup)
    Created - Dec 2016, WHT
    """

    #Get input (C or G)
    netIn,netInfo = process_input(netIn,['C','G','TO'])


    if D=='default' and netInfo['nettype'][0] == 'b':
        print('Default distance funciton specified. As network is binary, using Hamming')
        D='hamming'
    elif D=='default' and netInfo['nettype'][0] == 'w':
        D='euclidean'
        print('Default distance funciton specified. As network is weighted, using Euclidean')

    if isinstance(D,str) == False:
        raise ValueError('Distance metric must be a string')

    #If not directional, only do on the uppertriangle
    if netInfo['nettype'][1] == 'd':
        ind=np.triu_indices(netIn.shape[0],k=-netIn.shape[0])
    elif netInfo['nettype'][1] == 'u':
        ind=np.triu_indices(netIn.shape[0],k=1)

    #Get chosen distance metric fucntion
    distanceMetric=getDistanceFunction(D)

    if do=='global':
        V=np.mean([distanceMetric(netIn[ind[0],ind[1],t],netIn[ind[0],ind[1],t+1]) for t in range(0,netIn.shape[-1]-1)])
    if do=='time':
        V=[distanceMetric(netIn[ind[0],ind[1],t],netIn[ind[0],ind[1],t+1]) for t in range(0,netIn.shape[-1]-1)]
    #This takes quite a bit of time to loop through. When calculating per edge/node.
    if do=='edge' or do=='node':
        V = np.zeros([netIn.shape[0],netIn.shape[1]])
        for i in ind[0]:
            for j in ind[1]:
                V[i,j]=np.mean([distanceMetric(netIn[i,j,t],netIn[i,j,t+1]) for t in range(0,netIn.shape[-1]-1)])
        if netInfo['nettype'][1] == 'u':
            V = V + np.transpose(V)
        if do=='node':
            V = np.sum(V,axis=1)
    return V
