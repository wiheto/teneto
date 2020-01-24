import networkx as nx
from ..classes import TemporalNetwork
from ..utils import get_network_when
import numpy as np


def from_networkx(nxlist, output='TN'):
    """
    Creates a list of networkx objects for each slice

    Parameters
    -----------
    nxlist :  list
        A list of ordered NetworkX graphs. Each are a snapshot (assumes all nodes are present in all snapshots)
    output : str
        either: TN or array. Outputs either a TenetoNetwork object or numpy array.

    Returns
    tnet : tnetobject or array

    Examples
    ----------

    >>> import teneto
    >>> import numpy as np
    >>> import networkx as nx

    Create 3 random NetworkX graphs with 5 nodes.

    >>> np.random.seed(111)
    >>> G1 = nx.random_graphs.barabasi_albert_graph(5,3)
    >>> G2 = nx.random_graphs.barabasi_albert_graph(5,2)
    >>> G3 = nx.random_graphs.barabasi_albert_graph(5,4)

    Now we create a TenetoNetwork object from a list of networkx objects.

    >>> tnet = teneto.io.from_networkx([G1, G2, G3])
    >>> print(tnet.netshape)
    (5, 3)

    The multiple snapshots from NetworkX is now in Teneto!

    """
    tnet = [nx.to_numpy_array(n) for n in nxlist]
    tnet = np.stack(tnet).transpose([1, 2, 0])
    if output == 'TN':
        tnet = TemporalNetwork(from_array=tnet)
    return tnet
