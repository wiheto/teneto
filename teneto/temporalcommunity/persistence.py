import numpy as np

def persistence(communities, calc='global'):
    """
    Persistence is the proportion of consecutive time-points that a temporal community is in the same community at the next time-point

    Parameters
    ----------
    communities : array
        temporal communities of type: node,time (singlelabel) or node,node,time (for multilabel) communities

    calc : str
        can be 'global', 'time', or 'node'

    Returns
    --------
    persit_coeff : array
        the percentage of nodes that calculate the overall persistence (calc=global), or each node (calc=node), or for each time-point (calc=time)

    References
    -------
    Bazzi, Marya, et al. "Community detection in temporal multilayer networks, with an application to correlation networks." Multiscale Modeling & Simulation 14.1 (2016): 1-41.

    Note
    -----
    Bazzi et al present a non-normalized version with the global output.

    """

    reshape = False
    if len(communities.shape) == 3:
        ind = np.triu_indices(communities.shape[0], k=1)
        communities = communities[ind[0], ind[1], :]
        reshape = True

    if calc == 'global':
        persit_coeff = np.mean(communities[:, :-1] == communities[:, 1:])
    elif calc == 'node':
        if reshape:
            nnodes = len(np.unique(ind))
            persit_coeff = np.zeros(nnodes)
            for n in range(nnodes):
                i = np.where((ind[0] == n) | (ind[1] == n))[0]
                persit_coeff[n] = np.mean(communities[i, :-1] == communities[i, 1:])
        else:
            persit_coeff = np.mean(
                communities[:, :-1] == communities[:, 1:], axis=-1)

    elif calc == 'time':
        persit_coeff = np.hstack(
            [np.nan, np.mean(communities[:, :-1] == communities[:, 1:], axis=0)])
    return persit_coeff
