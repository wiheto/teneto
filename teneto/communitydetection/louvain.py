import community
import pandas as pd
import numpy as np
from scipy.spatial.distance import  jaccard
import networkx as nx
from ..utils import process_input, create_supraadjacency_matrix, tnet_to_nx, clean_community_indexes
from ..classes import TemporalNetwork

def temporal_louvain(tnet, resolution=1, intersliceweight=1, n_iter=100, negativeedge='ignore', randomseed=None, consensus_threshold=0.75, temporal_concsensus=True):
    r"""
    Louvain clustering for a temporal network

    Parameters 
    -----------
    tnet : array, dict, TemporalNetwork
        Input network
    resolution : int 
        resolution of Louvain clustering ($\gamma$)
    interslice : int
        interslice weight of multilayer clustering ($\omega$). Must be positive. 
    n_iter : int
        Number of iterations to run louvain for
    randomseed : int 
        Set for reproduceability 
    negativeedge : str
        If there are negative edges, what should be done with them. 
        Options: 'ignore' (i.e. set to 0). More options to be added. 
    consensus : float 
        When creating consensus matrix to average over number of iterations, keep values when the consensus is this amount. 

    Returns 
    -------
    communities : array (node,time)
        node,time array of community assignment

    Notes
    -------

    References
    ----------
    """

    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')
    supranet = create_supraadjacency_matrix(
        tnet, intersliceweight=intersliceweight)
    if negativeedge == 'ignore':
        supranet = supranet[supranet['weight'] > 0]
    nxsupra = tnet_to_nx(supranet)
    np.random.seed(randomseed)
    while True: 
        comtmp = np.zeros([tnet.N*tnet.T, n_iter]) - 1
        for n in range(n_iter):
            com = community.best_partition(nxsupra, resolution=resolution, randomize=True)
            comtmp[np.array(list(com.keys()), dtype=int), n] = list(com.values())
        comtmp = np.reshape(comtmp, [tnet.N, tnet.T, n_iter], order='F')
        nxsupra_old = nxsupra
        nxsupra = make_consensus_matrix(comtmp, consensus_threshold)
        if (nx.to_numpy_array(nxsupra) == nx.to_numpy_array(nxsupra_old)).all():
            break
    # TODO Add temporal consensus (greedy jaccard)
    communities = comtmp[:, :, 0]
    if temporal_concsensus == True: 
        communities = make_temporal_consensus(communities)
    return communities


def make_consensus_matrix(com_membership,th=0.5):
    r"""
    Makes the consensus matrix
.
    Parameters
    ----------

    com_membership : array
        Shape should be node, time, iteration.

    th : float
        threshold to cancel noisey edges

    Returns
    -------

    D : array
        consensus matrix
    """

    # So the question is whether this is applied perslice or not. 
    com_membership = np.array(com_membership)
    D = []
    for i in range(com_membership.shape[0]):
        for j in range(i+1, com_membership.shape[0]):
            con = np.sum((com_membership[i,:] - com_membership[j,:])==0, axis=-1) / com_membership.shape[-1]
            twhere = np.where(con > th)[0]
            D += list(zip(*[np.repeat(i,len(twhere)).tolist(),np.repeat(j,len(twhere)).tolist(),twhere.tolist(), con[twhere].tolist()]))

    D = pd.DataFrame(D, columns=['i', 'j', 't', 'weight'])
    D = TemporalNetwork(from_df=D)
    D = create_supraadjacency_matrix(D, intersliceweight=0)
    Dnx = tnet_to_nx(D)
    return Dnx


def make_temporal_consensus(com_membership):
    r"""
    Matches community labels accross time-points

    Jaccard matching is in a greedy fashiong. Matching the largest community at t with the community at t-1.

    Parameters
    ----------

    com_membership : array
        Shape should be node, time.

    Returns
    -------

    D : array
        temporal consensus matrix using Jaccard distance

    """

    com_membership = np.array(com_membership)
    D = []
    # make first indicies be between 0 and 1. 
    com_membership[:,0] = clean_community_indexes(com_membership[:,0])
    # loop over all timepoints, get jacccard distance in greedy manner for largest community to time period before
    for t in range(1, com_membership.shape[1]):
        ct, counts_t = np.unique(com_membership[:,t], return_counts=True)
        ct = ct[np.argsort(counts_t)[::-1]]
        c1back = np.unique(com_membership[:,t-1])
        new_index = np.zeros(com_membership.shape[0])
        bestcom = []
        for n in ct:
            if len(c1back) > 0:
                d = np.ones(int(c1back.max())+1)
                for m in c1back: 
                    v1 = np.zeros(com_membership.shape[0])
                    v2 = np.zeros(com_membership.shape[0])
                    v1[com_membership[:,t] == n]  = 1                
                    v2[com_membership[:,t-1] == m] = 1
                    d[int(m)] = jaccard(v1, v2)
                bestval = np.argmin(d)
            else: 
                bestval = new_index.max() + 1
            new_index[com_membership[:,t] == n] = bestval
            c1back = np.array(np.delete(c1back, np.where(c1back==bestval)))
        com_membership[:,t] = new_index
    return com_membership
