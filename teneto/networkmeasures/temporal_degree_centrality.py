"""Calculates temporal degree centrality"""

import numpy as np
from ..utils import process_input


def temporal_degree_centrality(tnet, axis=0, calc='overtime', communities=None,
                               decay=0, ignorediagonal=True):
    r"""
    Temporal degree of network.

    The sum of all connections each node has through time
    (either per timepoint or over the entire temporal sequence).

    Parameters
    -----------

    net : array, dict
        Temporal network input (graphlet or contact). Can have nettype: 'bu', 'bd', 'wu', 'wd'
    axis : int
        Dimension that is returned 0 or 1 (default 0).
        Note, only relevant for directed networks.
        i.e. if 0, node i has Aijt summed over j and t.
        and if 1, node j has Aijt summed over i and t.
    calc : str
        Can be following alternatives:

        'overtime' : returns a 1 x node vector. Returns the degree/stregnth over all time points.

        'pertime' : returns a node x time array. Returns the degree/strength per time point.

        'module_degree_zscore' : returns the Z-scored within community degree centrality
        (communities argument required). This is done for each time-point
        i.e. 'pertime' returns static degree centrality per time-point.
    ignorediagonal: bool
        if True, diagonal is made to 0.
    communities : array (Nx1)
        Vector of community assignment.
        If this is given and calc='pertime', then the strength within and
        between each communities is returned.
        (Note, this is not technically degree centrality).
    decay : int
        if calc = 'pertime', then decay is possible where the centrality of
        the previous time point is carried over to the next time point but decays
        at a value of $e^decay$ such that $D_d(t+1) = e^{-decay}D_d(t) + D(t+1)$.
        If decay is 0 then the final D will equal D when calc='overtime',
        if decay = inf then this will equal calc='pertime'.

    Returns
    ---------

    D : array
        temporal degree centrality (nodal measure).
        Array is 1D ('overtime'), 2D ('pertime', 'module_degree_zscore'),
        or 3D ('pertime' + communities (non-nodal/community measures)).


    Notes
    ------

    When the network is weighted, this could also be called "temporal strength"
    or "temporal strength centrality".
    This is a simple extension of the static definition.
    At times this has been defined slightly differently.
    Here we followed the definitions in [Degree-1]_ or [Degree-2]_.
    There are however many authors prior to this that have used temporal degree centrality.

    There are two basic versions of temporal degree centrality implemented:
    the average temporal degree centrality (``calc='overtime'``)
    and temporal degree centrality (``calc='pertime'``).

    When ``calc='pertime'``:

    .. math:: D_{it} = \sum_j A_{ijt}

    where A is the multi-layer connectivity matrix of the temporal network.

    This entails that :math:`D_{it}` is the sum of a node i's degree/strength at t.
    This has also been called the instantaneous degree centrality [Degree-2]_.

    When ``calc='overtime'``:

    .. math:: D_{i} = \sum_t\sum_j A_{ijt}

    i.e. :math:`D_{i}` is the sum of a node i's degree/strength over all time points.

    There are some additional options which can modify the estimate.
    One way is to add a decay term.
    This entails that ..math::`D_{it}`, uses some of the previous time-points estimate.
    An exponential decay is used here.

    .. math:: D_{it} = e^{-b} D_{i(t-1)} + \sum_j A_{ijt}

    where b is the deay parameter specified in the function.
    This, to my knowledge, was first introdueced by [Degree-2]_.

    References
    -----------

    .. [Degree-1]

        Thompson, et al (2017). From static to temporal network theory:
        Applications to functional brain connectivity.
        Network Neuroscience, 1(2), 69-99.
        [`Link <https://www.mitpressjournals.org/doi/full/10.1162/netn_a_00011>`_]

    .. [Degree-2]

        Masuda, N., & Lambiotte, R. (2016). A Guidance to Temporal Networks.
        [`Link to book's publisher
        <https://www.worldscientific.com/doi/abs/10.1142/9781786341150_0001>`_]

    """
    # Get input in right format
    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')
    if axis == 1:
        fromax = 'j'
        toax = 'i'
    else:
        fromax = 'i'
        toax = 'j'
    if tnet.sparse and tnet.nettype[0] == 'b':
        tnet.network['weight'] = 1
    # Diagonal is currently deleted.
    # if ignorediagonal:
    #     tnet = set_diagonal(tnet, 0)
    # sum sum tnet
    if calc == 'pertime' and communities is None:
        # Return node,time
        if tnet.sparse:
            tdeg = np.zeros([tnet.netshape[0], tnet.netshape[1]])
            df = tnet.network.groupby([fromax, 't']).sum().reset_index()
            tdeg[df[fromax], df['t']] = df['weight']
            # If undirected, do reverse
            if tnet.nettype[1] == 'u':
                df = tnet.network.groupby([toax, 't']).sum().reset_index()
                tdeg[df[toax], df['t']] += df['weight']
        else:
            tdeg = np.sum(tnet.network, axis=axis)
    elif calc == 'module_degree_zscore' and communities is None:
        raise ValueError(
            'Communities must be specified when calculating module degree z-score.')
    elif calc != 'pertime' and communities is None:
        # Return node
        if tnet.sparse:
            tdeg = np.zeros([tnet.netshape[0]])
            # Strength if weighted
            df = tnet.network.groupby([fromax])['weight'].sum().reset_index()
            tdeg[df[fromax]] += df['weight']
            # If undirected, do reverse
            if tnet.nettype[1] == 'u':
                df = tnet.network.groupby([toax])['weight'].sum().reset_index()
                tdeg[df[toax]] += df['weight']
        else:
            tdeg = np.sum(np.sum(tnet.network, axis=-1), axis=axis)
    elif calc == 'module_degree_zscore' and communities is not None:
        tdeg = np.zeros([tnet.netshape[0], tnet.netshape[1]])
        # Need to make this fully sparse
        if tnet.sparse:
            network = tnet.df_to_array()
        else:
            network = tnet.network
        for t in range(tnet.netshape[1]):
            if len(communities.shape) == 2:
                coms = communities[:, t]
            else:
                coms = communities
            for c in np.unique(coms):
                k_i = np.sum(network[
                             :, coms == c, t][coms == c], axis=axis)
                tdeg[coms == c, t] = (k_i - np.mean(k_i)) / np.std(k_i)
        tdeg[np.isnan(tdeg) == 1] = 0
    elif calc == 'pertime' and communities is not None:
        # neet to make this fully sparse
        if tnet.sparse:
            network = tnet.df_to_array()
        else:
            network = tnet.network
        tdeg_communities = np.zeros(
            [communities.max() + 1, communities.max() + 1, communities.shape[-1]])
        if len(communities.shape) == 2:
            for t in range(len(communities[-1])):
                coms = communities[:, t]
                unique_communities = np.unique(coms)
                for s1 in unique_communities:
                    for s2 in unique_communities:
                        tdeg_communities[s1, s2, t] = np.sum(
                            np.sum(network[coms == s1, :, t][:, coms == s2], axis=1), axis=0)
        else:
            unique_communities = np.unique(communities)
            tdeg_communities = [np.sum(np.sum(network[communities == s1][:, communities == s2],
                                              axis=1), axis=0)
                                for s1 in unique_communities for s2 in unique_communities]

        tdeg = np.array(tdeg_communities)
        tdeg = np.reshape(tdeg, [len(np.unique(communities)), len(
            np.unique(communities)), tnet.netshape[-1]])
        # Divide diagonal by 2 if undirected to correct for edges being present twice
        if tnet.nettype[1] == 'u':
            for s in range(tdeg.shape[0]):
                tdeg[s, s, :] = tdeg[s, s, :] / 2
    else:
        raise ValueError("invalid calc argument")

    if decay > 0 and calc == 'pertime':
        # Reshape so that time is first dimensions
        tdeg = tdeg.transpose(
            np.hstack([len(tdeg.shape) - 1, np.arange(len(tdeg.shape) - 1)]))
        for n in range(1, tdeg.shape[0]):
            tdeg[n] = np.exp(0 - decay) * tdeg[n - 1] + tdeg[n]
        tdeg = tdeg.transpose(np.hstack([np.arange(1, len(tdeg.shape)), 0]))
    elif decay > 0:
        print('WARNING: decay cannot be applied unless calc=time, ignoring decay')

    return tdeg
