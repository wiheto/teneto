import numpy as np

def promiscuity(communities):
    """
    Promiscuity calculates the number of communities each node is a member of.
    0 entails only 1 community. 1 entails all communities.

    Parameters
    ---------
    communities : array
        temporal communities labels of type (node,time). Temporal communities labels should be non-trivial through snapshots (i.e. temporal consensus clustering should be run)

    Returns
    -------
    promiscuity_coeff : array
        promiscuity of each node

    References
    ---------
    Papadopoulos, Lia, et al. "Evolution of network architecture in a granular material under compression." Physical Review E 94.3 (2016): 032908.

    """

    promiscuity_coeff = np.zeros(communities.shape[0])
    ncoms = len(np.unique(communities)) - 1
    for n in range(communities.shape[0]):
        promiscuity_coeff[n] = (len(np.unique(communities[n])) - 1) / ncoms
    return promiscuity_coeff