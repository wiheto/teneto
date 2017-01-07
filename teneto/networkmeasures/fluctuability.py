
import numpy as np
from teneto.utils import *

def fluctuability(netIn,do='global'):
    """
    fluctuability of temporal networks.This is the variation in unique edges through time divided by the overall number of edges.

    **PARAMETERS**

    :netIn: temporal network input (graphlet or contact)

        :nettype: 'bd', 'bu', 'wu', 'wd'

    :do: version of fluctuabiility to calcualte. 'global' (i.e. average distance of all nodes for each consecutive time point). A nodal version may be added in future.

    **OUTPUT**

    :F: Fluctuability

        :format: scalar (do='global')

    **SEE ALSO**
    - *voalitility*

    **HISTORY**

    :Modified: Jan 2016, WHT (documentation)
    :Created: Dec 2016, WHT

    """

    #Get input type (C or G)
    netIn,netInfo = process_input(netIn,['C','G','TO'])

    netIn[netIn!=0]=1
    U=np.sum(netIn,axis=2)
    U[U>0]=1
    U[U==0]=0

    F = (np.sum(U)) / np.sum(netIn)
    return F
