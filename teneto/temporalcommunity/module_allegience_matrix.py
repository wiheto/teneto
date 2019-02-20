import numpy as np

def module_allegience_matrix(community):
    """
    Computes the module allegiance matrix with values representing the probability that
    nodes i and j were assigned to the same community by time-varying clustering methods.

    parameters
    ----------
    community : array
        array of community assignment of size node,time

    returns
    -------
    P : array
        module allegiance matrix, with P_ij probability that area i and j are in the same community

    Reference:
    ----------
    Bassett, et al. (2013) “Robust detection of dynamic community structure in networks”, Chaos, 23, 1

    """
    N = len(community)
    C = np.unique(community)
    T = P = np.zeros([N, N])

    for t in range(len(community[0, :])):
        for i in range(len(community[:, 0])):
            for j in range(len(community[:, 0])):
                if i == j:
                    continue
                # T_ij indicates the number of times that i and j are assigned to the same community across time
                if community[i][t] == community[j][t]:
                    T[i, j] += 1

    # module allegiance matrix, probability that ij were assigned to the same community
    P = (1/C)*T

    return P
