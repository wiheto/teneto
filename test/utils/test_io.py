import pandas as pd
import networkx as nx
import teneto


def test_tnet_to_nx():
    df = pd.DataFrame({'i': [0, 0], 'j': [1, 2], 't': [0, 1]})
    dfnx = teneto.utils.tnet_to_nx(df, t=0)
    G = nx.to_numpy_array(dfnx)
    if not G.shape == (2, 2):
        raise AssertionError()
    if not G[0, 1] == 1:
        raise AssertionError()
