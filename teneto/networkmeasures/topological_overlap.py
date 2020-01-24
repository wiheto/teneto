"""Calculates topological overlap"""
import numpy as np
from ..utils import process_input


def topological_overlap(tnet, calc='pertime'):
    u"""
    Topological overlap quantifies the persistency of edges through time.

    If two consequtive time-points have similar edges, this becomes high (max 1).
    If there is high change, this becomes 0.

    References: [topo-1]_, [topo-2]_

    Parameters
    ----------
    tnet : array, dict
        graphlet or contact sequence input. Nettype: 'bu'.
    calc: str
        which version of topological overlap to calculate:
        'node' - calculates for each node, averaging over time.
        'pertime' - (default) calculates for each node per time points.
        'overtime' - calculates for each node per time points.


    Returns
    -------
    topo_overlap : array
        if calc = 'pertime', array is (node,time) in size.
        if calc = 'node', array is (node) in size.
        if calc = 'overtime', array is (1) in size. The final time point returns as nan.

    Notes
    ------
    When edges persist over time, the topological overlap increases.
    It can be calculated as a global valu, per node, per node-time.

    When calc='pertime', then the topological overlap is:

    .. math::

        TopoOverlap_{i,t} = {\sum_j G_{i,j,t} G_{i,j,t+1}
        \over \sqrt{\sum_j G_{i,j,t} \sum_j G_{i,j,t+1}}}

    When calc='node', then the topological overlap is the mean of math:`TopoOverlap_{i,t}`:

    .. math:: AvgTopoOverlap_{i} = {1 \over T-1} \sum_t TopoOverlap_{i,t}

    where T is the number of time-points.
    This is called the *average topological overlap*.

    When calc='overtime', the *temporal-correlation coefficient* is calculated

    .. math:: TempCorrCoeff = {1 \over N} \sum_i AvgTopoOverlap_i

    where N is the number of nodes.

    For all the three measures above, the value is between 0 and 1 where 0
    entails "all edges changes" and 1 entails "no edges change".


    Examples
    ---------

    First import all necessary packages

    >>> import teneto
    >>> import numpy as np

    Then make an temporal network with 3 nodes and 4 time-points.

    >>> G = np.zeros([3, 3, 3])
    >>> i_ind = np.array([0, 0, 0, 0,])
    >>> j_ind = np.array([1, 1, 1, 2,])
    >>> t_ind = np.array([0, 1, 2, 2,])
    >>> G[i_ind, j_ind, t_ind] = 1
    >>> G = G + G.transpose([1,0,2]) # Make symmetric

    Now the topological overlap can be calculated:

    >>> topo_overlap = teneto.networkmeasures.topological_overlap(G)

    This returns *topo_overlap* which is a (node,time) array.
    Looking above at how we defined G,
    when t = 0, there is only the edge (0,1).
    When t = 1, this edge still remains.
    This means topo_overlap should equal 1 for node 0 at t=0 and 0 for node 2:

    >>> topo_overlap[0,0]
    1.0
    >>> topo_overlap[2,0]
    0.0

    At t=2, there is now also an edge between (0,2),
    this means node 0's topological overlap at t=1 decreases as
    its edges have decreased in their persistency at the next time point
    (i.e. some change has occured). It equals ca. 0.71

    >>> topo_overlap[0,1]
    0.7071067811865475

    If we want the average topological overlap, we simply add the calc argument to be 'node'.

    >>> avg_topo_overlap = teneto.networkmeasures.topological_overlap(G, calc='node')

    Now this is an array with a length of 3 (one per node).

    >>> avg_topo_overlap
    array([0.85355339, 1.        , 0.        ])

    Here we see that node 1 had all its connections persist, node 2 had no connections persisting, and node 0 was in between.

    To calculate the temporal correlation coefficient,

    >>> temp_corr_coeff = teneto.networkmeasures.topological_overlap(G, calc='overtime')

    This produces one value reflecting all of G

    >>> temp_corr_coeff
    0.617851130197758


    References
    ----------
    .. [topo-1]

        Tang et al (2010) Small-world behavior in time-varying graphs.
        Phys. Rev. E 81, 055101(R) [`arxiv link <https://arxiv.org/pdf/0909.1712.pdf>`_]
    .. [topo-2]

        Nicosia et al (2013) "Graph Metrics for Temporal Networks"
        In: Holme P., Saramäki J. (eds) Temporal Networks.
        Understanding Complex Systems. Springer.
        [`arxiv link <https://arxiv.org/pdf/1306.0493.pdf>`_]

    """
    tnet = process_input(tnet, ['C', 'G', 'TN'])[0]

    numerator = np.sum(tnet[:, :, :-1] * tnet[:, :, 1:], axis=1)
    denominator = np.sqrt(
        np.sum(tnet[:, :, :-1], axis=1) * np.sum(tnet[:, :, 1:], axis=1))

    topo_overlap = numerator / denominator
    topo_overlap[np.isnan(topo_overlap)] = 0

    if calc == 'pertime':
        # Add missing timepoint as nan to end of time series
        topo_overlap = np.hstack(
            [topo_overlap, np.zeros([topo_overlap.shape[0], 1])*np.nan])
    else:
        topo_overlap = np.mean(topo_overlap, axis=1)
        if calc == 'node':
            pass
        elif calc == 'overtime':
            topo_overlap = np.mean(topo_overlap)

    return topo_overlap
