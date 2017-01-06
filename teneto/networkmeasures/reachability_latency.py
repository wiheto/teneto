import numpy as np
from teneto.networkmeasures.shortest_temporal_path import shortest_temporal_path

"""

Reachability algorithem.

"""


def reachability_latency(datIn,r=1,do='global'):
    """
    returns global reachability latency.
    This is the r-th longest temporal path. Where r is the number of time If r=1,

    **PARAMETERS**

    :datIn: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu'

        :paths: Dictionary of paths (output of shortest_temporal_path).

    :r: reachability ratio that the latency is calculated in relation to.
        Value must be over 0 and up to 1.
        1 (default) - all nodes must be reached.
        Other values (e.g. .5 imply that 50% of nodes are reached)
        This is rounded to the nearest node inter.
        E.g. if there are 6 nodes [1,2,3,4,5,6], it will be node 4 (due to round upwards)

    :do: what to calculate R for:
        :'global': entire network.
        :'nodes': each node.


    **OUTPUT**

    :R: readability latency
        :format: integer (numpy array)

    **SEE ALSO**

    - *temporal_efficiency*
    - *shortest_temporal_path*

    **HISTORY**

    :Modified: Dec 2016, WHT (Documentation)
    :Created: Dec 2016, WHT

    """


    sp=0 #are shortest paths calculated
    if isinstance(datIn,dict):
        #This could be done better
        if [k for k in list(datIn.keys()) if k=='paths']==['paths']:
            sp=1
    # if shortest paths are not calculated, calculate them
    if sp==0:
        datIn = shortest_temporal_path(datIn)

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
