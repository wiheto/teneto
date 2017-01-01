import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path

"""

Temporal Efficiency

"""


def temporal_efficiency(datIn):
    """
    returns temporal efficiency estimate.


    **PARAMETERS**

    :datIn: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu'

        :paths: Dictionary of paths (output of shortest_temporal_path).


    **OUTPUT**

    :E: global temporal efficiency (global measure)

        :format: integer (numpy array)

    **NOTES**

    This can be implemented on a non-global level in the future.

    **SEE ALSO**

    - *shortesttemporalpath*

    **HISTORY**

    Modified - Jan 2016, WHT (documentation)
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
