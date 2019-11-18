"""Generatenetwork a random poisson network"""

import numpy as np
from ..utils import graphlet2contact, set_diagonal


def rand_poisson(nnodes, ncontacts, lam=1, nettype='bu', netinfo=None, netrep='graphlet'):
    """
    Generate a random network where intervals between contacts are distributed by a poisson distribution

    Parameters
    ----------

    nnodes : int
        Number of nodes in networks

    ncontacts : int or list
        Number of expected contacts (i.e. edges). If list, number of contacts for each node.
        Any zeros drawn are ignored so returned degree of network can be smaller than ncontacts.

    lam : int or list
        Expectation of interval.

    nettype : str
        'bu' or 'bd'

    netinfo : dict
        Dictionary of additional information

    netrep : str
        How the output should be.

    If ncontacts is a list, so should lam.

    Returns
    -------
        net : array or dict
            Random network with intervals between active edges being Poisson distributed.

    """
    if isinstance(ncontacts, list):
        if len(ncontacts) != nnodes:
            raise ValueError(
                'Number of contacts, if a list, should be one per node')
    if isinstance(lam, list):
        if len(lam) != nnodes:
            raise ValueError(
                'Lambda value of Poisson distribution, if a list, should be one per node')
    if isinstance(lam, list) and not isinstance(ncontacts, list) or not isinstance(lam, list) and isinstance(ncontacts, list):
        raise ValueError(
            'When one of lambda or ncontacts is given as a list, the other argument must also be a list.')

    if nettype == 'bu':
        edgen = int((nnodes*(nnodes-1))/2)
    elif nettype == 'bd':
        edgen = int(nnodes*nnodes)

    if not isinstance(lam, list) and not isinstance(ncontacts, list):
        icts = np.random.poisson(lam, size=(edgen, ncontacts))
        net = np.zeros([edgen, icts.sum(axis=1).max()+1])
        for n in range(edgen):
            net[n, np.unique(np.cumsum(icts[n]))] = 1
    else:
        icts = []
        ict_max = 0
        for n in range(edgen):
            icts.append(np.random.poisson(lam[n], size=ncontacts[n]))
            if sum(icts[-1]) > ict_max:
                ict_max = sum(icts[-1])
        net = np.zeros([nnodes, ict_max+1])
        for n in range(nnodes):
            net[n, np.unique(np.cumsum(icts[n]))] = 1

    if nettype == 'bu':
        nettmp = np.zeros([nnodes, nnodes, net.shape[-1]])
        ind = np.triu_indices(nnodes, k=1)
        nettmp[ind[0], ind[1], :] = net
        net = nettmp + nettmp.transpose([1, 0, 2])
    elif nettype == 'bd':
        net = net.reshape([nnodes, nnodes, net.shape[-1]], order='F')
        net = set_diagonal(net, 0)

    if netrep == 'contact':
        if not netinfo:
            netinfo = {}
        netinfo['nettype'] = 'b' + nettype[-1]
        net = graphlet2contact(net, netinfo)

    return net
