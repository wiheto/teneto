import numpy as np
from ..utils import process_input

def topological_overlap(tnet, calc='time'): 
    """
    Topological overlap quantifies the persistency of edges through time. If two consequtive time-points have similar edges, this becomes high (max 1). If there is high change, this becomes 0. 
    
    References: [topo-1], [topo-2]

    Parameters
    ----------
    tnet : array, dict
        graphlet or contact sequence input. Nettype: 'bu'.
    calc: str 
        which version of topological overlap to calculate: 
        'node' - calculates for each node, averaging over time. 
        'time' - (default) calculates for each node per time points.
        'global' - (default) calculates for each node per time points.


    Returns
    -------
    topo_overlap : array 
        if calc = 'time', array is (node,time) in size.
        if calc = 'node', array is (node) in size. 
        if calc = 'global', array is (1) in size. 

    Notes 
    ------
    When edges persist over time, the topological overlap increases. It can be calculated as a global valu, per node, per node-time. 

    When calc='time', then the topological overlap is:   

    .. math:: TopoOverlap_{i,t} =  {\sum_j G_{i,j,t} G_{i,j,(t+1)} \over \sqrt{\sum_j G{i,j,t} \sum_j G{i,j,t}}}

    When calc='node', then the topological overlap is the mean of math:`TopoOverlap_{i,t}`:   

    .. math:: \text{AvgTopoOverlap}_{i} = {1 \over T-1} \sum_t TopoOverlap

    where T is the number of time-points. This is called the _average topological overlap_.

    When calc='node', the *temporal-correlation coefficient* is calculated

    .. math:: \text{TempCorrCoeff}_{i,t} = \sum_j {G_{i,j,t} \over G_{i,j,(t+1)}}


    References
    ----------
    .. [topo-1] Tang et al (2010) Small-world behavior in time-varying graphs. Phys. Rev. E 81, 055101(R) [`arxiv link <https://arxiv.org/pdf/0909.1712.pdf>`_]
    .. [topo-2] Nicosia et al (2013) "Graph Metrics for Temporal Networks" In: Holme P., Saramäki J. (eds) Temporal Networks. Understanding Complex Systems. Springer. 
        [`arxiv link <https://arxiv.org/pdf/1306.0493.pdf>`_]

    """

    tnet = process_input(tnet, ['C', 'G', 'TO'])[0]
    
    numerator = np.sum(tnet[:,:,:-1] * tnet[:,:,1:],axis=1)
    denominator = np.sqrt(np.sum(tnet[:,:,:-1],axis=1) * np.sum(tnet[:,:,1:],axis=1))

    topo_overlap = numerator / denominator
    topo_overlap[np.isnan(topo_overlap)] = 0

    if calc == 'time': 
        # Add missing timepoint as nan to end of time series
        topo_overlap = np.hstack([topo_overlap,np.zeros([topo_overlap.shape[0],1])*np.nan])
    else: 
        topo_overlap = np.mean(topo_overlap,axis=1)    
        if calc == 'node': 
            pass
        elif calc == 'global': 
            topo_overlap = np.mean(topo_overlap)

    return topo_overlap