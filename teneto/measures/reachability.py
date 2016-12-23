import numpy as np
from teneto.measures.shortesttemporalpath import temporalPaths

"""

Reachability algorithem.

"""


def reachabilityLatency(datIn,r=1,do='global'):
    """
    returns global reachability latency.
    This is the r-th longest temporal path. Where r is the number of time If r=1,

    Parameters
    ----------
    datIn: Variable input which can be
        A) Temporal graph of format (can be bu):
            (i) G: graphlet (3D numpy array).
            (ii) C: contact (dictionary)
        B) Dictionary of paths (output of temporalPath function).

    r: reachability ratio that the latency is calculated in relation to.
        Value must be over 0 and up to 1.
        1 (default) - all nodes must be reached.
        Other values (e.g. .5 imply that 50% of nodes are reached)
        This is rounded to the nearest node inter.
        E.g. if there are 6 nodes [1,2,3,4,5,6], it will be node 4 (due to round upwards)

    do: 'global' or 'nodes'


    Returns
    ----------
    R, readability latency
    format: integer (numpy array)

    See Also
    ----------
    temporalEfficiency
    shortesttemporalpath

    History
    ----------
    Created - Dec 2016, WHT
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


    rArg=netShape[0]-np.round(netShape[0]*r)

    R = np.zeros([netShape[1],netShape[2]])*np.nan
    for t in range(0,netShape[2]):
        s=-np.sort(-datIn['paths'][:,:,t],axis=1)
        R[:,t] = s[:,rArg]
    if do == 'global':
        R = np.nansum(R)
        R = R/((netShape[0])*netShape[2])
    elif do == 'nodes':
        R = np.nansum(R,axis=1)
        R = R/(netShape[2])
    return R
