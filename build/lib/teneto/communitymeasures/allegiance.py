import numpy as np


def allegiance(community):
    u"""
    Computes allience of communities.

    The allegiance matrix with values representing the probability that
    nodes i and j were assigned to the same community by time-varying clustering methods.[alleg-1]_

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

    .. [alleg-1]:

        Bassett, et al. (2013)
        “Robust detection of dynamic community structure in networks”, Chaos, 23, 1

    """
    N = community.shape[0]
    C = community.shape[1]
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
    # Make diagonal nan
    np.fill_diagonal(P, np.nan)
    return P
