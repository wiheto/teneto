"""Calculates temporal betweenness centrality"""

import numpy as np
from .shortest_temporal_path import shortest_temporal_path


def temporal_betweenness_centrality(tnet=None, paths=None, calc='pertime'):
    r"""
    Returns temporal betweenness centrality per node.

    Parameters
    -----------

    data : array or dict

        Temporal network input (graphlet or contact). nettype: 'bu', 'bd'.

    calc : str

        either 'overtime' or 'pertime'

    paths : pandas dataframe

        Output of TenetoBIDS.networkmeasure.shortest_temporal_paths

    Note
    -----

    Input should be *either* tnet or paths.


    Returns
    --------

    :close: array

        normalized temporal betweenness centrality.

            If calc = 'pertime', returns (node,time)

            If calc = 'overtime', returns (node)

    Notes
    --------

    Temporal betweenness centrality uses the shortest temporal
    paths and calculates betweennesss from it.

    Teneto returns a normalized betweenness centrality value,
    defined as [Bet-1]_:

    .. math::

        B_{it} = {1 \over (N-1)(N-2)} \sum_{j = 1; j \neq i}
        \sum_{k = 1; k \neq i,j} {\sigma^i_{jkt} \over \sigma_{jk}}

    If there is a shortest temporal path from j to k, starting at t that
    goes through node i, then :math:`\sigma^i_{jkt}` is 1, otherwise 0.
    :math:`\sigma_{jk}` is the total number of paths that exist from j to k.
    The remaining part of the equation normalizes by the number of nodes.

    If a temporal network is used as input (i.e. not the paths), then teneto
    uses :py:func:`.shortest_temporal_path` to calculates the shortest paths.
    See :py:func:`.shortest_temporal_path` for more details.

    If ``calc=overtime`` then the average B over time is returned.

    References
    ---------

    .. [Bet-1]

        Tang, J., Musolesi, M., Mascolo, C., Latora, V., & Nicosia, V. (2010).
        Analysing Information Flows and Key Mediators through Temporal Centrality
        Metrics Categories and Subject Descriptors.
        Proceedings of the 3rd Workshop on Social Network Systems.
        [`Link https://doi.org/10.1145/1852658.1852661`_]

    """
    if tnet is not None and paths is not None:
        raise ValueError('Only network or path input allowed.')
    if tnet is None and paths is None:
        raise ValueError('No input.')
    # if shortest paths are not calculated, calculate them
    if tnet is not None:
        paths = shortest_temporal_path(tnet)

    bet = np.zeros([paths[['from', 'to']].max().max() +
                    1, paths['t_start'].max() + 1])

    for row in paths.iterrows():
        if (np.isnan(row[1]['path includes'])).all():
            pass
        else:
            nodes_in_path = np.unique(np.concatenate(
                row[1]['path includes'])).astype(int).tolist()
            nodes_in_path.remove(row[1]['from'])
            nodes_in_path.remove(row[1]['to'])
            sigmajk = paths[(paths['from'] == row[1]['from']) & (paths['to'] == row[1]['to'])]
            sigmajk = sigmajk.dropna(subset=['temporal-distance'])
            sigmajk = len(sigmajk)
            if len(nodes_in_path) > 0:
                bet[nodes_in_path, row[1]['t_start']] += 1 / sigmajk

    # Normalise bet
    bet = (1 / ((bet.shape[0] - 1) * (bet.shape[0] - 2))) * bet

    if calc == 'overtime':
        bet = np.mean(bet, axis=1)

    return bet
