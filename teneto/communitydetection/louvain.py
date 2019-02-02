import community
import pandas as pd
import numpy as np
from ..utils import process_input
from ..utils import tnet_to_nx
from ..utils import create_supraadjacency_matrix


def temporal_louvain(tnet, resolution=1, intersliceweight=1, n_iter=100, negativeedge='ignore', randomseed=None, consensus_threshold=0.75):
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
    return communities


def make_consensus_matrix(com_membership,th=0.5):
    r"""
    Makes the consensus matrix
.
    Parameters
    ----------

    com_membership : array
        Shape should be node, iteration.

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
    D = teneto.TemporalNetwork(from_df=D)
    D = create_supraadjacency_matrix(D, intersliceweight=0)
    Dnx = tnet_to_nx(D)
    return Dnx
