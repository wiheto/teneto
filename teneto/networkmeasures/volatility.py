from teneto.utils import *
import numpy as np

def volatility(netIn,D='default',do='global',subnetworkID=[]):
    """
    volatility of temporal networks. This is the average distance between consecutive time points of graphlets (difference is caclualted either globally, per edge)

    **PARAMETERS**

    :netIn: temporal network input (graphlet or contact)

        :netIn nettype: 'bu','bd','wu','wd'

    :D: distance function. Following options available: 'default', 'hamming', 'euclidean'. (Default implies hamming for binary networks, euclidean for weighted).
    :do: version of volaitility to caclulate. Possibilities include:

        :'global': (default): the average distance of all nodes for each consecutive time point).
        :'edge': average distance between consecutive time points for each edge). Takes considerably longer
        :'node': (i.e. returns the average per node output when calculating volatility per 'edge').
        :'time': returns volatility per time point
        :'subnetwork': returns volatility per subnetwork id (see subnetworkID). Also is returned per time-point and this may be changed in the future (with additional options)
	:'subnetworkID': vector of integers.
        :Note: Index of subnetworks are returned "as is" with a shape of [max(subnetworks)+1,max(subnetworks)+1]. So if the indexes used are [1,2,3,5], V.shape==(6,6). The returning V[1,2] will correspond indexes 1 and 2. And missing index (e.g. here 0 and 4 will be NANs in rows and columns). If this behaviour is unwanted, call clean_subnetwork_indexes first.

    :network:

    **OUTPUT**

    :V: Volatility

        :format: scalar (do='global'),
            1d numpy array (do='node'),
            2d numpy array (do='edge')

    **SEE ALSO**
    - *utils.getDistanceFunction*

    **HISTORY**
    Modified - Dec 2016, WHT (documentation, cleanup)
    Created - Dec 2016, WHT
    """

    #Get input (C or G)
    netIn,netInfo = process_input(netIn,['C','G','TO'])


    if D=='default' and netInfo['nettype'][0] == 'b':
        print('Default distance funciton specified. As network is binary, using Hamming')
        D='hamming'
    elif D=='default' and netInfo['nettype'][0] == 'w':
        D='euclidean'
        print('Default distance funciton specified. As network is weighted, using Euclidean')

    if isinstance(D,str) == False:
        raise ValueError('Distance metric must be a string')

    #If not directional, only do on the uppertriangle
    if netInfo['nettype'][1] == 'd':
        ind=np.triu_indices(netIn.shape[0],k=-netIn.shape[0])
    elif netInfo['nettype'][1] == 'u':
        ind=np.triu_indices(netIn.shape[0],k=1)

    if do=='subnetwork':
        #Make sure subnetworkID is np array for indexing later on.
        subnetworkID = np.array(subnetworkID)
        if len(subnetworkID)!=netInfo['netshape'][0]:
            raise ValueError('When processing per network, subnetworkID vector must equal the number of nodes')
        if subnetworkID.min()<0:
            raise ValueError('Subnetwork assignments must be positive integers')


    #Get chosen distance metric fucntion
    distanceMetric=getDistanceFunction(D)


    if do=='global':
        V=np.mean([distanceMetric(netIn[ind[0],ind[1],t],netIn[ind[0],ind[1],t+1]) for t in range(0,netIn.shape[-1]-1)])
    elif do=='time':
        V=[distanceMetric(netIn[ind[0],ind[1],t],netIn[ind[0],ind[1],t+1]) for t in range(0,netIn.shape[-1]-1)]
    #This takes quite a bit of time to loop through. When calculating per edge/node.
    elif do=='edge' or do=='node':
        V = np.zeros([netIn.shape[0],netIn.shape[1]])
        for i in ind[0]:
            for j in ind[1]:
                V[i,j]=np.mean([distanceMetric(netIn[i,j,t],netIn[i,j,t+1]) for t in range(0,netIn.shape[-1]-1)])
        if netInfo['nettype'][1] == 'u':
            V = V + np.transpose(V)
        if do=='node':
            V = np.sum(V,axis=1)
    elif do == 'subnetwork':
        netIDs = set(subnetworkID)
        V = np.zeros([max(netIDs)+1,max(netIDs)+1,netInfo['netshape'][-1]-1])
        for net1 in netIDs:
            for net2 in netIDs:
                Vtmp=[distanceMetric(netIn[subnetworkID==net1][:,subnetworkID==net2,t   ],netIn[subnetworkID==net1][:,subnetworkID==net2,t+1]) for t in range(0,netIn.shape[-1]-1)]
                V[net1,net2,:]=Vtmp
    elif do == 'withinsubnetwork':
        within_ind = np.array([[ind[0][n], ind[1][n]] for n in range(0,len(ind[0])) if subnetworkID[ind[0][n]] == subnetworkID[ind[1][n]]])
        V=[distanceMetric(netIn[within_ind[:,0],within_ind[:,1],t],netIn[within_ind[:,0],within_ind[:,1],t+1]) for t in range(0,netIn.shape[-1]-1)]
    elif do == 'betweensubnetwork':
        between_ind = np.array([[ind[0][n], ind[1][n]] for n in range(0,len(ind[0])) if subnetworkID[ind[0][n]] != subnetworkID[ind[1][n]]])
        V=[distanceMetric(netIn[between_ind[:,0],between_ind[:,1],t],netIn[between_ind[:,0],between_ind[:,1],t+1]) for t in range(0,netIn.shape[-1]-1)]

    return V
