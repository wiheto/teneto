import numpy as np
from teneto.utils import *


def shortest_temporal_path(netIn,q=1):
    """
    Calculates the shortest temporal path when all possible routes cam be travelled at each time point.
    Currently only works for binary undirected edges (but can be expanded).

    **PARAMETERS**

    :netIn: temporal network input (graphlet or contact)

        :nettype: 'bu'

    :q: quiet (default = 1). Turn to 0 if you want progree update.

    **OUTPUT**

    :paths: shortest temporal paths

        :format: dictionary

    Paths are of the struction, path['paths'][i,j,t] - shortest path for i to reach j, starting at time t.

    **NOTE**

    This function assumes all paths can be taken per time point.
    In a future update, this function temporalPaths will allow for only a portion of edges to be travelled per time point.
    This will be implmeneted with no change to the funcitonality of calling this function as it is today, with the defaults being all edges can be travelled.


    **SEE ALSO**

    - *temporal_efficiency*
    - *reachability_latency*
    - *temporal_closeness_centrality*

    **HISTORY**

    Modified - Dec 2016, WHT (documentation)
    Created - Nov 2016, WHT

    """

    #Get input type (C or G)
    #Process input
    netIn,netInfo = process_input(netIn,['C','G','TO'])


    if netInfo['nettype'] != 'bu':
        raise ValueError('It looks like your graph is not binary and undirected. Shortest temporal paths can only be calculated for binary undirected networks in Teneto at the moment. If another type is required, please create an issue at github.com/wiheto/teneto and I will try and prioritize this.')


    #Preallocate output
    P = np.zeros(netInfo['netshape'])*np.nan
    #Go backwards in time and see if something is reached
    P_last = np.zeros([netInfo['netshape'][0],netInfo['netshape'][1]])*np.nan
    for t in list(reversed(range(0,netInfo['netshape'][2]))):
        if q==0:
            print('--- Running for time: ' + str(t) + ' ---')
        fid = np.where(netIn[:,:,t]>=1)
        #Update time step
        #Note to self: Add a conditional to prevent nan warning that can pop out if no path is there straight away.
        P_last+=1
        #Reset connections present to 1s
        P_last[fid[0],fid[1]]=1
        # Update nodes with no connections
        # Nodes to update are nodes with an edge present at the time point
        for v in np.unique(fid[0]):
            a=np.where(P_last[v,:]==1)[0]
            #P_last_preupdate = np.array(P_last[v,:])
            P_last[v,:]=np.nanmin(P_last[np.hstack([a,v]),:],axis=0)
            # P_last[np.hstack([a,v]),:][:,np.where(P_last[v,:]<P_last_preupdate)[0]]
            # for n in np.where(P_last[v,:]<P_last_preupdate)[0]: #These nodes are updated.
            #
            #np.where(P_last[np.hstack([a,v]),:]<=np.nanmin(P_last[np.hstack([a,v]),:],axis=0))[0]
            P_last[v,v]=np.nan # make self connection nan regardless
        P[:,:,t]=P_last
    ## Return output
    paths = {}
    r=(P.size-np.sum(np.isnan(P)))/(P.size-(P.shape[0]*P.shape[2]))
    paths['percentReached']=r
    paths['paths']=P
    paths['nettype']=netInfo['nettype']

    return paths
