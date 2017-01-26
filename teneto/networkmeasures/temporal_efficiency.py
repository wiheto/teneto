import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path

"""

Temporal Efficiency

"""


def temporal_efficiency(datIn,do='global'):
    """
    returns temporal efficiency estimate.


    **PARAMETERS**

    :datIn: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu'

        :paths: Dictionary of paths (output of shortest_temporal_path).


    :do: 'global' (default) - measure averages over time and nodes.
        'node' or 'node_from' average over nodes (i) and time. Giving average efficiency for i to j.
        'node_to' measure average over nodes j and time. Giving average efficiency using paths to j from  i.


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
        datIn = shortest_temporal_path(datIn)

    # Calculate efficiency which is 1 over the mean path.
    if do == 'global':
        E=1/np.nanmean(datIn['paths'])
    elif do == 'node' or do == 'node_from':
        E=1/np.nanmean(np.nanmean(datIn['paths'],axis=2),axis=1)
    elif do == 'node_to':
        E=1/np.nanmean(np.nanmean(datIn['paths'],axis=2),axis=0)

    return E
