"""Export temporalnetwork to other software"""

import networkx as nx
from ..utils import process_input, create_supraadjacency_matrix


def to_networkx(tnet, export_type='supra', t=None, ):
    """
    Creates a list of networkx objects for each slice

    Parameters
    -----------
    tnet :  array, dict, tnetobject
        Temporal network
    export_type : str
        either: supra or snapshot.
        This either export a networkx objects for the entire supraadjacency matrix
        (all timepoints) or one object per snapshot in a list.
    t : int
        if export_type=='snapshot', you can specify a single time point or multiple.

    Returns
    nxobj : list or networkxobject

    Examples
    ----------

    >>> import teneto
    >>> import numpy as np
    >>> import networkx as nx

    Create a binary matrix that is 4 nodes and 3 time points with a probability of 0.5 that there is an edge.

    >>> np.random.seed(111)
    >>> tnet = teneto.generatenetwork.rand_binomial([4,3],[0.5])

    Now we create the supraadjacency networkx object. Note how

    >>> supranet = teneto.io.to_networkx(tnet)
    >>> print(supranet.number_of_nodes())
    12
    >>> print(supranet.number_of_edges())
    16
    >>> print(tnet.sum())
    16.0

    Alternatively, it is possible to create a networkx object for each

    >>> nxlist = teneto.io.to_networkx(tnet, 'snapshot')
    >>> print(len(nxlist))
    3
    >>> print(nxlist[0].number_of_nodes())
    4

    Note
    ---------

    If you want to create networkx.MultiGraph you have to do it yourself.

    """
    tnet = process_input(tnet, ['C', 'G', 'TN'], 'G')[0]
    if export_type == 'supra':
        tnet = create_supraadjacency_matrix(tnet).sort_values(['i', 'j'])
        nxobj = nx.from_pandas_edgelist(
            tnet, source='i', target='j', edge_attr='weight')
    elif export_type == 'snapshot':
        if t is None:
            t = range(0, tnet.shape[-1])
        nxobj = []
        for i in t:
            nxobj.append(nx.from_numpy_array(tnet[:, :, i]))
    return nxobj
