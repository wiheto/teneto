import numpy as np
from teneto.utils import *


def temporalPaths(netIn,q=1):
    """
    Calculates the shortest temporal path when all possible routes cam be travelled at each time point.
    Currently only works for binary undirected edges (but can be expanded).

    Parameters:
    ----------
    netIn: Temporal graph of format (can be bu only at present):
        (i) G: graphlet (3D numpy array).
        (ii) C: contact (dictionary)
    q: quiet (default = 1). Turn to 0 if you want progree update.

    Returns
    ----------
    temporal degree (centrality measure)
    format: 1d numpy array

    Note
    ----------
    This function assumes all paths can be taken per time point.
    In a future update, this function temporalPaths will allow for only a portion of edges to be travelled per time point.
    This will be implmeneted with no change to the funcitonality of calling this function as it is today, with the defaults being all edges can be travelled.


    See Also
    ----------

    History
    ----------
    Created - Nov 2016, WHT

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

    if nettype != 'bu':
        raise ValueError('It looks like your graph is not binary and undirected. Shortest temporal paths can only be calculated for binary undirected networks in Teneto at the moment. If another type is required, please create an issue at github.com/wiheto/teneto and I will try and prioritize this.')


    #Preallocate output
    P = np.zeros(netIn.shape)*np.nan
    #Go backwards in time and see if something is reached
    P_last = np.zeros([netIn.shape[0],netIn.shape[1]])*np.nan
    for t in list(reversed(range(0,netIn.shape[2]))):
        if q==0:
            print('--- Running for time: ' + str(t) + ' ---')
        fid = np.where(netIn[:,:,t]>=1)
        #Update time step
        #Note to self: And a conditional to prevent nan warning that can pop out if no path is there straight away.
        P_last+=1
        #Reset connections present to 1
        P_last[fid[0],fid[1]]=1
        # Update nodes with no connections
        # Nodes to update are nodes with an edge present at the time point
        for v in fid[0]:
            a=np.where(P_last[v,:]==1)[0]
            P_last[v,:]=np.nanmin(P_last[np.hstack([a,v]),:],axis=0)
            P_last[v,v]=np.nan # make self connection nan regardless
        P[:,:,t]=P_last
    ## Return output
    out = {}
    r=(P.size-np.sum(np.isnan(P)))/(P.size-(P.shape[0]*P.shape[2]))
    out['percentReached']=r
    out['paths']=P
    out['nettype']=nettype

    return out
