import numpy as np
from teneto.measures.shortesttemporalpath import temporalPaths

"""

Temporal degree algorithems.

"""


def temporalCloseness(datIn,d=0):
    """
    returns temporal closeness centrality per node.
    As temporalPaths only works with binary undirected edges at the moment, this is required for temporal closeness centrality.

    Parameters
    ----------
    datIn: Variable input which can be
        A) Temporal graph of format (can be bu):
            (i) G: graphlet (3D numpy array).
            (ii) C: contact (dictionary)
        B) Dictionary of paths (output of temporalPath function).


    Returns
    ----------
    temporal closness (centrality measure)
    format: 1d numpy array

    See Also
    ----------
    temporalPaths
    temporalDegree

    History
    ----------
    Created - Nov 2016, WHT
    """

    sp=0 #are shortest paths calculated
    if isinstance(datIn,dict):
        #This could be done better
        if [k for k in list(datIn.keys()) if k=='paths']==['paths']:
            sp=1
    # if shortest paths are not calculated, calculate them
    if sp==0:
        datIn = temporalPaths(datIn)

    netShape = datIn['paths'].shape


    C=np.nansum(1/np.nanmean(datIn['paths'],axis=2),axis=1)/(datIn['paths'].shape[1]-1)


    return C
