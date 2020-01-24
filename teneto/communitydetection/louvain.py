import community
import pandas as pd
import numpy as np
from scipy.spatial.distance import jaccard
import networkx as nx
from teneto.utils import process_input, create_supraadjacency_matrix, tnet_to_nx, clean_community_indexes
from teneto.classes import TemporalNetwork
from concurrent.futures import ProcessPoolExecutor, as_completed


def temporal_louvain(tnet, resolution=1, intersliceweight=1, n_iter=100,
                     negativeedge='ignore', randomseed=None, consensus_threshold=0.5,
                     temporal_consensus=True, njobs=1):
    r"""
    Louvain clustering for a temporal network.

    Parameters
    -----------
    tnet : array, dict, TemporalNetwork
        Input network
    resolution : int
        resolution of Louvain clustering ($\gamma$)
    intersliceweight : int
        interslice weight of multilayer clustering ($\omega$). Must be positive.
    n_iter : int
        Number of iterations to run louvain for
    randomseed : int
        Set for reproduceability
    negativeedge : str
        If there are negative edges, what should be done with them.
        Options: 'ignore' (i.e. set to 0). More options to be added.
    consensus : float (0.5 default)
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
    # Divide resolution by the number of timepoints
    #resolution = resolution / tnet.T
    supranet = create_supraadjacency_matrix(
        tnet, intersliceweight=intersliceweight)
    if negativeedge == 'ignore':
        supranet = supranet[supranet['weight'] > 0]
    nxsupra = tnet_to_nx(supranet)
    np.random.seed(randomseed)
    i = 0
    while True:
        print(i)
        i += 1
        comtmp = []
        if njobs > 1:
            with ProcessPoolExecutor(max_workers=njobs) as executor:
                job = {executor.submit(
                    _run_louvain, nxsupra, resolution, tnet.N, tnet.T) for n in range(n_iter)}
                for j in as_completed(job):
                    comtmp.append(j.result())
            comtmp = np.stack(comtmp)
        else:
            comtmp = np.array(
                [_run_louvain(nxsupra, resolution, tnet.N, tnet.T) for n in range(n_iter)])
        comtmp = np.stack(comtmp)
        comtmp = comtmp.transpose()
        comtmp = np.reshape(comtmp, [tnet.N, tnet.T, n_iter], order='F')
        # if n_iter == 1:
        #    break
        nxsupra_old = nxsupra
        nxsupra = make_consensus_matrix(comtmp, consensus_threshold)
        # If there was no consensus, there are no communities possible, return
        if nxsupra is None:
            break
        if (nx.to_numpy_array(nxsupra, nodelist=np.arange(tnet.N*tnet.T)) == nx.to_numpy_array(nxsupra_old, nodelist=np.arange(tnet.N*tnet.T))).all():
            break
    communities = comtmp[:, :, 0]
    if temporal_consensus:
        communities = make_temporal_consensus(communities)
    return communities


def _run_louvain(nxsupra, resolution, N, T):
    comtmp = np.zeros([N*T])
    com = community.best_partition(
        nxsupra, resolution=resolution, random_state=None)
    comtmp[np.array(list(com.keys()), dtype=int)] = list(com.values())
    return comtmp


def make_consensus_matrix(com_membership, th=0.5):
    r"""
    Makes the consensus matrix.

    From multiple iterations, finds a consensus partition.
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
    com_membership = np.array(com_membership)
    D = []
    for i in range(com_membership.shape[0]):
        for j in range(i+1, com_membership.shape[0]):
            con = np.sum((com_membership[i, :] - com_membership[j, :])
                         == 0, axis=-1) / com_membership.shape[-1]
            twhere = np.where(con > th)[0]
            D += list(zip(*[np.repeat(i, len(twhere)).tolist(), np.repeat(j,
                                                                          len(twhere)).tolist(), twhere.tolist(), con[twhere].tolist()]))
    if len(D) > 0:
        D = pd.DataFrame(D, columns=['i', 'j', 't', 'weight'])
        D = TemporalNetwork(from_df=D)
        D = create_supraadjacency_matrix(D, intersliceweight=0)
        Dnx = tnet_to_nx(D)
    else:
        Dnx = None
    print(D)
    return Dnx


def make_temporal_consensus(com_membership):
    r"""
    Matches community labels accross time-points.

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
    # make first indicies be between 0 and 1.
    com_membership[:, 0] = clean_community_indexes(com_membership[:, 0])
    # loop over all timepoints, get jacccard distance in greedy manner for largest community to time period before
    for t in range(1, com_membership.shape[1]):
        ct, counts_t = np.unique(com_membership[:, t], return_counts=True)
        ct = ct[np.argsort(counts_t)[::-1]]
        c1back = np.unique(com_membership[:, t-1])
        new_index = np.zeros(com_membership.shape[0])
        for n in ct:
            if len(c1back) > 0:
                d = np.ones(int(c1back.max())+1)
                for m in c1back:
                    v1 = np.zeros(com_membership.shape[0])
                    v2 = np.zeros(com_membership.shape[0])
                    v1[com_membership[:, t] == n] = 1
                    v2[com_membership[:, t-1] == m] = 1
                    d[int(m)] = jaccard(v1, v2)
                bestval = np.argmin(d)
            else:
                bestval = new_index.max() + 1
            new_index[com_membership[:, t] == n] = bestval
            c1back = np.array(np.delete(c1back, np.where(c1back == bestval)))
        com_membership[:, t] = new_index
    return com_membership
