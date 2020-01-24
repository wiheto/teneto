import networkx as nx
from .utils import get_network_when


def tnet_to_nx(df, t=None):
    """Creates undirected networkx object"""
    if t is not None:
        df = get_network_when(df, t=t)
    if 'weight' in df.columns:
        nxobj = nx.from_pandas_edgelist(
            df, source='i', target='j', edge_attr='weight')
    else:
        nxobj = nx.from_pandas_edgelist(df, source='i', target='j')
    return nxobj