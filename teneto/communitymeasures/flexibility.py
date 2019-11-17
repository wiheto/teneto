import numpy as np


def flexibility(communities):
    """
    Amount a node changes community

    Parameters
    ----------
    communities : array
        Community array of shape (node,time)

    Returns
    --------
    flex : array
        Size with the flexibility of each node.

    Notes
    -----
    Flexbility calculates the number of times a node switches its community label during a time series [flex-1]_.
    It is normalized by the number of possible changes which could occur.
    It is important to make sure that the different community labels accross time points are not artbirary.

    References
    -----------

    .. [flex-1]

        Bassett, DS, Wymbs N, Porter MA, Mucha P, Carlson JM, Grafton ST.
        Dynamic reconfiguration of human brain networks during learning.
        PNAS, 2011, 108(18):7641-6.
    """
    # Preallocate
    flex = np.zeros(communities.shape[0])
    # Go from the second time point to last, compare with time-point before
    for t in range(1, communities.shape[1]):
        flex[communities[:, t] != communities[:, t-1]] += 1
    # Normalize
    flex = flex / (communities.shape[1] - 1)
    return flex
