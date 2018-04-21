
import igraph as ig
import louvain
from scipy.spatial.distance import jaccard
import itertools
import numpy as np
import teneto

def temporal_louvain_with_consensus(net, iter_n=100, resolution_parameter=1, interslice_weight=0, quality_function='NewmanGirvan2004', seed=42, consensus_threshold=0.5):
    """
    Temporal louvain clustering run for iter_n times and consenus matrix returned.

    Parameters
    ----------
    net : array, dict
        network representation (contact sequences or graphlet)
    iter_n : int
        nummber of repeated louvain clustering
    resolution_parameter : int
        Spatial resolution parameter. Only valid for some qualtiy functions. Default=1.
        Resolution parameter is only needed for ReichardtBornholdt2006, and
    interslice_weight : int
        The weight that connects the different graphlets/snapshots to eachother. Default=0
    quality function : str
        What type of louvain clustering is done. Options: NewmanGirvan2004, TraagVanDoorenNesterov2011, ReichardtBornholdt2006
    seed : int
        Seed for reproduceability
    consensus_threshold : float
        Value between 0 and 1. When creating consensus matrix, ignore if value only occurs in specified fraction of iterations. If 0.5 two nodes must be in the same community 50% of the time to be considered in the consensus matrix.

    Returns
    -------
    communities : array, dict
        Consensus matrix from louvain clustering. Dimensions: commiunities, time.
        Dict is returned (contact representation) if input is contact representation.

    Qualtify funciton sources
    --------------------------

    NewmanGirvan2004 :
        Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure in networks. Physical Review E, 69(2), 026113. 10.1103/PhysRevE.69.026113
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#modularityvertexpartition>`_
    ReichardtBornholdt2006 :
        Reichardt, J., & Bornholdt, S. (2006). Statistical mechanics of community detection. Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#rbconfigurationvertexpartition>`_
    TraagVanDoorenNesterov2011 :
        Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011). Narrow scope for resolution-limit-free community detection. Physical Review E, 84(1), 016114. 10.1103/PhysRevE.84.016114
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#cpmvertexpartition>`_
    TraagKringsVanDooren2013 :
        Traag, V. A., Krings, G., & Van Dooren, P. (2013). Significant scales in community structure. Scientific Reports, 3, 2930. 10.1038/srep02930
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#significancevertexpartition>`_
    TraagAldecoaDelvenne2015 :
        Traag, V. A., Aldecoa, R., & Delvenne, J.-C. (2015). Detecting communities using asymptotical surprise. Physical Review E, 92(2), 022816. 10.1103/PhysRevE.92.022816
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#surprisevertexpartition>`_

    Dependencies
    ------------
    These functions make use of iGraph (http://igraph.org/python/) and louvain-igraph (http://louvain-igraph.readthedocs.io/en/latest/)

    Note
    ----
    At the moment input should generally only be positive edges.

    """
    if isinstance(net,dict):
        dict_input = True
    else:
        dict_input = False
    net, netinfo = teneto.utils.process_input(net, ['C', 'G', 'TO'])

    Gin = net.copy()
    D = np.random.random(np.prod(Gin.shape)).reshape(Gin.shape)
    i=0
    # While condition doesn't seem to be run at the moment
    while len(np.where((D > 0) & (D < 1))[0]) > 0:
        i=i+1
        C = teneto.communitydetection.temporal_louvain(Gin, iter_n=iter_n, resolution_parameter=resolution_parameter, seed=seed, interslice_weight=interslice_weight, quality_function=quality_function)
        D = [teneto.communitydetection.make_consensus_matrix(C[:,:,t],th=consensus_threshold) for t in range(C.shape[-1])]
        D = np.dstack(D)
        Gin = D
        #Only first iteration needs to be returned as consensus means they are all identical
    communities = teneto.communitydetection.make_temporal_consensus(C[0,:,:])
    if dict_input:
        C = teneto.utils.graphlet2contact(net,netinfo)
        C['communities'] = communities
        return C
    else:
        return communities

def temporal_louvain(net, iter_n=1, resolution_parameter=1, interslice_weight=1, quality_function='NewmanGirvan2004', seed=100):
    """
    Temporal louvain clustering run for iter_n times.

    Parameters
    ----------
    net : array, dict
        network representation (contact sequences or graphlet)
    iter_n : int
        nummber of repeated louvain clustering
    resolution_parameter : int
        Spatial resolution parameter. Only valid for some qualtiy functions. Default=1.
        Resolution parameter is only needed for ReichardtBornholdt2006, and
    interslice_weight : int
        The weight that connects the different graphlets/snapshots to eachother. Default=0
    quality function : str
        What type of louvain clustering is done. Options: NewmanGirvan2004, TraagVanDoorenNesterov2011, ReichardtBornholdt2006
    seed : int
        Seed for reproduceability
    consensus_threshold : float
        Value between 0 and 1. When creating consensus matrix, ignore if value only occurs in specified fraction of iterations. If 0.5 two nodes must be in the same community 50% of the time to be considered in the consensus matrix.

    Returns
    -------
    communities : array
        Louvain clustering. Dimensions: [iter_n], commiunities, time

    Qualtify funciton sources
    --------------------------

    NewmanGirvan2004 :
        Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure in networks. Physical Review E, 69(2), 026113. 10.1103/PhysRevE.69.026113
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#modularityvertexpartition>`_
    ReichardtBornholdt2006 :
        Reichardt, J., & Bornholdt, S. (2006). Statistical mechanics of community detection. Physical Review E, 74(1), 016110. 10.1103/PhysRevE.74.016110
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#rbconfigurationvertexpartition>`_
    TraagVanDoorenNesterov2011 :
        Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011). Narrow scope for resolution-limit-free community detection. Physical Review E, 84(1), 016114. 10.1103/PhysRevE.84.016114
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#cpmvertexpartition>`_
    TraagKringsVanDooren2013 :
        Traag, V. A., Krings, G., & Van Dooren, P. (2013). Significant scales in community structure. Scientific Reports, 3, 2930. 10.1038/srep02930
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#significancevertexpartition>`_
    TraagAldecoaDelvenne2015 :
        Traag, V. A., Aldecoa, R., & Delvenne, J.-C. (2015). Detecting communities using asymptotical surprise. Physical Review E, 92(2), 022816. 10.1103/PhysRevE.92.022816
        `Read more <http://louvain-igraph.readthedocs.io/en/latest/reference.html#surprisevertexpartition>`_

    Dependencies
    ------------
    These functions make use of iGraph (http://igraph.org/python/) and louvain-igraph (http://louvain-igraph.readthedocs.io/en/latest/)

    Note
    ----
    At the moment input should generally only be positive edges.



    """
    if isinstance(net,dict):
        dict_input = True
    else:
        dict_input = False
    net, netinfo = teneto.utils.process_input(net, ['C', 'G', 'TO'])
    if quality_function == 'TraagVanDoorenNesterov2011':
        louvain_alg = louvain.CPMVertexPartition
        louvain_kwags = {'resolution_parameter': resolution_parameter}
    elif quality_function == 'ReichardtBornholdt2006':
        louvain_alg = louvain.RBConfigurationVertexPartition
        louvain_kwags = {'resolution_parameter': resolution_parameter}
    elif quality_function == 'NewmanGirvan2004':
        louvain_alg = louvain.ModularityVertexPartition
        louvain_kwags = {}
    elif quality_function == 'TraagKringsVanDooren2013':
        louvain_alg = louvain.SignificanceVertexPartition
        louvain_kwags = {}
    elif quality_function == 'TraagAldecoaDelvenne2015':
        louvain_alg = louvain.SurpriseVertexPartition
        louvain_kwags = {}
    g_to_ig = []
    for i in range(net.shape[-1]):
        g_to_ig.append(ig.Graph.Weighted_Adjacency(net[:,:,i].tolist()))
    for n in range(net.shape[0]):
        for t in range(net.shape[-1]):
            g_to_ig[t].vs[n]['id'] = n
    membership = []
    louvain.set_rng_seed(seed)
    if interslice_weight != 0:
        for n in range(0,iter_n):
            mem, improvement = louvain.find_partition_temporal(
                    g_to_ig,
                    louvain_alg,
                    interslice_weight=interslice_weight,
                    **louvain_kwags)
            membership.append(mem)
        com_membership = np.array(membership).transpose([0,2,1])
    else:
        com_membership = []
        for n in range(0,iter_n):
            membership = []
            for snapshot in g_to_ig:
                mem = louvain.find_partition(
                    snapshot,
                    louvain_alg,
                    **louvain_kwags)
                membership.append(mem.membership)
            com_membership.append(membership)
        com_membership = np.array(com_membership).transpose([0,2,1])

    if dict_input:
        C = teneto.utils.graphlet2contact(net,netinfo)
        C['communities'] = np.squeeze(com_membership)
        return C
    else:
        return np.squeeze(com_membership)


def make_consensus_matrix(com_membership,th=0.5):
    """
    Makes the consensus matrix
.
    Parameters
    ----------

    com_membership : array
        Shape should be iterations,node.

    th : float
        threshold to cancel noisey edges

    Returns
    -------

    D : array
        consensus matrix
    """
    com_membership = np.array(com_membership)
    D = np.zeros([com_membership.shape[1],com_membership.shape[1]])
    for it in range(com_membership.shape[0]):
        for c in np.unique(com_membership[it,:]):
            id = np.where(com_membership[it,:]==c)
            if len(id[0]) > 1:
                id = np.array(list(itertools.combinations(id[0],2)))
                D[id[:,0],id[:,1]] += 1
                D[id[:,1],id[:,0]] += 1
    D = D/com_membership.shape[0]
    D[D<th] = 0
    np.fill_diagonal(D,1)
    return D

def make_temporal_consensus(communities):
    """
    Links communities where communities have been derived at multiple time points.

    Parameters
    ---------
    """
    for t in range(1,communities.shape[-1]):
        community_possibilities = []
        jaccard_best = 1
        cidx = list(itertools.permutations(np.unique(communities[:,t]),len(np.unique(communities[:,t]))))
        for c in cidx:
            ctmp = communities[:,t].copy()
            for i,n in enumerate(c):
                ctmp[communities[:,t]==i]=n
            if jaccard(communities[:,t-1],ctmp) < jaccard_best:
                community_possibilities = ctmp
        communities[:,t] = community_possibilities
    return communities
