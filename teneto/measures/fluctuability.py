
import numpy as np
from teneto.utils import *

def fluctuability(netIn,D='hamming',do='global'):
    """
    fluctuability of temporal networks.

    This is the variation in unique edges through time divided by the overall number of edges.

    Parameters
    ----------
    netIn: Temporal graph of format (can be bd,bu,wu,wd):
        (i) G: graphlet (3D numpy array).
        (ii) C: contact (dictionary)

    do: version of fluctuabiility to calcualte:
        'global' (i.e. average distance of all nodes for each consecutive time point).
        --- a nodal version may be added but only global works ---

    Returns
    ----------
    Volatility
    format: scalar (do='global')

    See Also
    ----------

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

    netIn[netIn!=0]=1
    U=np.sum(netIn,axis=2)
    U[U>0]=1
    U[U==0]=0

    F = (np.sum(U)) / np.sum(netIn)
    return F
