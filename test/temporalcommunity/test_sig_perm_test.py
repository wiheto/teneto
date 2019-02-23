import community
import networkx as nx
import numpy as np
from sig_perm_test import sig_perm_test

def test_sig_perm_test():
    """

    See "Example of multiple community detection:"
        https://en.wikipedia.org/wiki/Modularity_(networks)

    """
    A = np.array([[0,1,1,0,0,0,0,0,0,1],
                  [1,0,1,0,0,0,0,0,0,0],
                  [1,1,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,1],
                  [0,0,0,1,0,1,0,0,0,0],
                  [0,0,0,1,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,1,1],
                  [0,0,0,0,0,0,1,0,1,0],
                  [0,0,0,0,0,0,1,1,0,0],
                  [1,0,0,1,0,0,1,0,0,0]])

    G = nx.Graph(A)

    part = community.best_partition(G)
    com = [part.get(node) for node in G.nodes()]

    com_d = {key : val for key,val in enumerate(com)}
    com_l = [{0,1,2,9},{3,4,5},{6,7,8}]

    Q1= community.modularity(com_d,G)
    Q2=nx.algorithms.community.modularity(G,com_l)

    # Q1 ≈ Q2

    T = 10000

    sig_matrix, q_matrix, q_matrix_r = sig_perm_test(net,community,T)

    # note:
    # modularity is normalized in nx.algorithms.community.modularity with norm = 1 / (2 * m) where m = G.size() (which equals np.sum(A)/2))

    m = np.sum(A) # == np.sum(q_matrix)

    Q = np.sum(q_matrix) * (1/(2*(m+1)))

    # Q1 ≈ Q2 ≈ Q = 0.48...
