import numpy as np
from teneto.measures.shortesttemporalpath import temporalPaths

"""

Temporal Efficiency

"""


def temporalEfficiency(datIn):
    """
    returns temporal efficiency estimate.


    Parameters
    ----------
    datIn: Variable input which can be
        A) Temporal graph of format (can be bu):
            (i) G: graphlet (3D numpy array).
            (ii) C: contact (dictionary)
        B) Dictionary of paths (output of temporalPath function).


    Returns
    ----------
    E, global temporal efficiency (global measure)
    format: integer (numpy array)

    NOTES
    ---------
    This can be implemented on a non-global level in the future.

    See Also
    ----------
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

    # Calculate efficiency which is 1 over the mean path.
    E=1/np.nanmean(datIn['paths'])

    return E
