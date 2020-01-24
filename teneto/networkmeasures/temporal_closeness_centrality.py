"""Calculates temporal closeness centrality"""

import numpy as np
from .shortest_temporal_path import shortest_temporal_path


def temporal_closeness_centrality(tnet=None, paths=None):
    r"""
    Returns temporal closeness centrality per node.

    Temporal closeness centrlaity is the sum of a node's
    average temporal paths with all other nodes.

    Parameters
    -----------

    tnet : array, dict, object

        Temporal network input with nettype: 'bu', 'bd'.

    paths : pandas dataframe

        Output of TenetoBIDS.networkmeasure.shortest_temporal_paths

    Note
    ------

    Only one input (tnet or paths) can be supplied to the function.

    Returns
    --------

    :close: array

        temporal closness centrality (nodal measure)

    Notes
    -------

    Temporal closeness centrality is defined in [Close-1]_:

    .. math:: C^T_{i} = {{1} \over {N-1}}\sum_j{1\over\\tau_{ij}}

    Where :math:`\\tau_{ij}` is the average temporal paths between node i and j.

    Note, there are multiple different types of temporal distance measures
    that can be used in temporal networks.
    If a temporal network is used as input (i.e. not the paths), then teneto
    uses :py:func:`.shortest_temporal_path` to calculates the shortest paths.
    See :py:func:`.shortest_temporal_path` for more details.

    .. [Close-1]

        Pan, R. K., & Saramäki, J. (2011).
        Path lengths, correlations, and centrality in temporal networks.
        Physical Review E - Statistical, Nonlinear, and Soft Matter Physics, 84(1).
        [`Link https://doi.org/10.1103/PhysRevE.84.016105`_]

    """
    if tnet is not None and paths is not None:
        raise ValueError('Only network or path input allowed.')
    if tnet is None and paths is None:
        raise ValueError('No input.')
    # if shortest paths are not calculated, calculate them
    if tnet is not None:
        paths = shortest_temporal_path(tnet)

    # Change for HDF5: paths.groupby([from,to])
    # Then put preallocated in a pathmat 2D array
    pathmat = np.zeros([paths[['from', 'to']].max().max() + 1,
                        paths[['from', 'to']].max().max() + 1,
                        paths[['t_start']].max().max() + 1]) * np.nan
    pathmat[paths['from'].values, paths['to'].values,
            paths['t_start'].values] = paths['temporal-distance']

    closeness = np.nansum(1 / np.nanmean(pathmat, axis=2),
                          axis=1) / (pathmat.shape[1] - 1)

    return closeness
