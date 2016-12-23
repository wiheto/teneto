from teneto.utils import *
import numpy as np

def volatility(netIn,D='default',do='global'):
    """
    volatility of temporal networks. This is the average distance between consecutive time points of graphlets (difference is caclualted either globally, per edge)

    Parameters
    ----------
    netIn: Temporal graph of format (can be bd,bu,wu,wd):
        (i) G: graphlet (3D numpy array).
        (ii) C: contact (dictionary)
    D: Distance metric to be used. Options:
        'default': if binary, 'hamming'; if weighted, 'euclidean'
        'hamming'
        'euclidean'
        'taxicab'
    do: version of volaitility to caclulate:
        'global' (default): the average distance of all nodes for each consecutive time point).
        'edge': average distance between consecutive time points for each edge). Takes considerably longer
        'node' (i.e. returns the average per node output when calculating volatility per 'edge').

    Returns
    ----------
    Volatility
    format: scalar (do='global'),
            1d numpy array (do='node'),
            2d numpy array (do='edge')

    See Also
    ----------
    utils.getDistanceFunction

    History
    ----------
    Created - Dec 2016, WHT
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


    if D=='default' and nettype[0] == 'b':
        print('Default distance funciton specified. As network is binary, using Hamming')
        D='hamming'
    elif D=='default' and nettype[0] == 'w':
        D='euclidean'
        print('Default distance funciton specified. As network is weighted, using Euclidean')

    if isinstance(D,str) == False:
        raise ValueError('Distance metric must be a string')

    #If not directional, only do on the uppertriangle
    if nettype[1] == 'd':
        ind=np.triu_indices(netIn.shape[0],k=-netIn.shape[0])
    elif nettype[1] == 'u':
        ind=np.triu_indices(netIn.shape[0],k=1)

    #Get chosen distance metric fucntion
    distanceMetric=getDistanceFunction(D)

    if do=='global':
        vol=np.mean([distanceMetric(netIn[ind[0],ind[1],t],netIn[ind[0],ind[1],t+1]) for t in range(0,netIn.shape[-1]-1)])
    #This takes quite a bit of time to loop through. When calculating per edge/node.
    if do=='edge' or do=='node':
        vol = np.zeros([netIn.shape[0],netIn.shape[1]])
        for i in ind[0]:
            for j in ind[1]:
                vol[i,j]=np.mean([distanceMetric(netIn[i,j,t],netIn[i,j,t+1]) for t in range(0,netIn.shape[-1]-1)])
        if nettype[1] == 'u':
            vol = vol + np.transpose(vol)
        if do=='node':
            vol = np.sum(vol,axis=1)
    return vol
