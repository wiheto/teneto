import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path



def temporal_closeness_centrality(datIn):
    '''
    returns temporal closeness centrality per node.
    As temporalPaths only works with binary undirected edges at the moment, this is required for temporal closeness centrality.

    **PARAMETERS**

    :datIn: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu'

        :paths: Dictionary of paths (output of shortest_temporal_path).


    **OUTPUT**

    :C: temporal closness centrality (nodal measure)

        :format: 1d numpy array

    **See Also**

    - *temporalPaths*
    - *temporalDegree*

    **History**

    Modified - Dec 2016, WHT (documentation, cleanup)
    Created - Nov 2016, WHT

    '''

    sp=0 #are shortest paths calculated
    if isinstance(datIn,dict):
        #This could be done better
        if [k for k in list(datIn.keys()) if k=='paths']==['paths']:
            sp=1
    # if shortest paths are not calculated, calculate them
    if sp==0:
        datIn = shortest_temporal_path(datIn)

    netShape = datIn['paths'].shape


    C=np.nansum(1/np.nanmean(datIn['paths'],axis=2),axis=1)/(datIn['paths'].shape[1]-1)


    return C
