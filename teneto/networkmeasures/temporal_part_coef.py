import numpy as np
from ..utils import process_input

def temporal_part_coef(tnet, communities=None, removeneg=False):
    r'''
    Temporal participation coefficient is a measure of diversity of connections across communities for individual nodes.

    Parameters
    ----------
    tnet : array, dict
        graphlet or contact sequence input. Only positive matrices considered.
    communities : array
        community vector. Either 1D (node) community index or 2D (node,time).
    removeneg : bool (default false)
        If true, all values < 0 are made to be 0. 


    Returns
    -------
    P : array
        participation coefficient


    Notes
    -----

    Static participatoin coefficient is:

    .. math:: P_i = 1 - \sum_s^{N_M}({{k_{is}}\over{k_i}})^2 

    Where s is the index of each community (:math:`N_M`). :math:`k_i` is total degree of node. And :math:`k_{is}` is degree of connections within community.[part-1]_

    This "temporal" version only loops through temporal snapshots and calculates :math:`P_i` for each t.

    If directed, function sums axis=1, so tnet may need to be transposed before hand depending on what type of directed part_coef you are interested in.


    References
    ----------

    .. [part-1] Guimera et al (2005) Functional cartography of complex metabolic networks. Nature. 433: 7028, p895-900. [`Link <http://doi.org/10.1038/nature03288>`_]
    '''

    if communities is None:
        if isinstance(tnet,dict):
            if 'communities' in tnet.keys():
                communities = tnet['communities']
            else:
                raise ValueError('Community index not found')
        else:
            raise ValueError('Community must be provided for graphlet input')

    # Get input in right format
    tnet, netinfo = process_input(tnet, ['C', 'G', 'TN'])

    if np.sum(tnet<0) > 0 and not removeneg:
        raise ValueError('Negative connections found')
    if removeneg:
        tnet[tnet<0] = 0

    k_is = np.zeros([netinfo['netshape'][0],netinfo['netshape'][2]])
    part = np.ones([netinfo['netshape'][0],netinfo['netshape'][2]])

    for t in np.arange(0,netinfo['netshape'][2]):
        if len(communities.shape)==2:
            C = communities[:,t]
        else:
            C = communities
        for i in np.unique(C):
            k_is[:,t] += np.square(np.sum(tnet[:,C == i,t], axis=1))

    part = part - (k_is / np.square(np.sum(tnet, axis=1)))
    # Set any division by 0 to 0
    part[np.isnan(part)==1] = 0

    return part
