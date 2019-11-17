import numpy as np
#from ..utils import process_input
from teneto.utils import process_input
import pandas as pd


def temporal_participation_coeff(tnet, communities=None, decay=None, removeneg=False):
    r"""
    Calculates the temporal participation coefficient

    Temporal participation coefficient is a measure of diversity of connections across communities for individual nodes.

    Parameters
    ----------
    tnet : array, dict
        graphlet or contact sequence input. Only positive matrices considered.
    communities : array
        community vector. Either 1D (node) community index or 2D (node,time).
    removeneg : bool (default false)
        If true, all values < 0 are made to be 0.


    Returns
    -------
    P : array
        participation coefficient


    Notes
    -----

    Static participatoin coefficient is:

    .. math:: P_i = 1 - \sum_s^{N_M}({{k_{is}}\over{k_i}})^2

    Where s is the index of each community (:math:`N_M`).
    :math:`k_i` is total degree of node.
    And :math:`k_{is}` is degree of connections within community.[part-1]_

    This "temporal" version only loops through temporal snapshots and calculates :math:`P_i` for each t.

    If directed, function sums axis=1,
    so tnet may need to be transposed before hand depending on what type of directed part_coef you are interested in.


    References
    ----------

    .. [part-1]

        Guimera et al (2005) Functional cartography of complex metabolic networks.
        Nature. 433: 7028, p895-900. [`Link <http://doi.org/10.1038/nature03288>`_]
    """
    if communities is None:
        if isinstance(tnet, dict):
            if 'communities' in tnet.keys():
                communities = tnet['communities']
            else:
                raise ValueError('Community index not found')
        else:
            raise ValueError('Community must be provided for graphlet input')

    # Get input in right format
    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')

    if tnet.nettype[0] == 'w':
        # TODO add contingency when hdf5 data has negative edges
        if not tnet.hdf5 and tnet.sparse:
            if sum(tnet.network['weight'] < 0) > 0 and not removeneg:
                print(
                    'TENETO WARNING: negative edges exist when calculating participation coefficient.')
            else:
                tnet.network['weight'][tnet.network['weight'] < 0] = 0
        if not tnet.hdf5 and not tnet.sparse:
            if np.sum(tnet.network< 0) > 0 and not removeneg:
                print(
                    'TENETO WARNING: negative edges exist when calculating participation coefficient.')
            else:
                tnet.network[tnet.network < 0] = 0


    part = np.zeros([tnet.netshape[0], tnet.netshape[1]])



    if len(communities.shape) == 1:
        for t in np.arange(0, tnet.netshape[1]):
            C = communities
            snapshot = tnet.get_network_when(t=t)
            if tnet.nettype[1] == 'd':
                i_at_t = snapshot['i'].values
            else:
                i_at_t = np.concatenate(
                    [snapshot['i'].values, snapshot['j'].values])
            i_at_t = np.unique(i_at_t).tolist()
            i_at_t = list(map(int, i_at_t))
            for i in i_at_t:
                # Calculate degree of node
                if tnet.nettype[1] == 'd':
                    df = tnet.get_network_when(i=i, t=t)
                    j_at_t = df['j'].values
                    if tnet.nettype == 'wd':
                        k_i = df['weight'].sum()
                    elif tnet.nettype == 'bd':
                        k_i = len(df)
                elif tnet.nettype[1] == 'u':
                    df = tnet.get_network_when(ij=i, t=t)
                    j_at_t = np.concatenate([df['i'].values, df['j'].values])
                    if tnet.nettype == 'wu':
                        k_i = df['weight'].sum()
                    elif tnet.nettype == 'bu':
                        k_i = len(df)
                j_at_t = list(map(int, j_at_t))
                for c in np.unique(C[j_at_t]):
                    ci = np.where(C == c)[0].tolist()
                    k_is = tnet.get_network_when(i=i, j=ci, t=t)
                    if tnet.nettype[1] == 'u' and tnet.sparse:
                        k_is2 = tnet.get_network_when(j=i, i=ci, t=t)
                        k_is = pd.concat([k_is, k_is2])
                    if len(k_is) > 0:
                        if tnet.nettype[0] == 'b':
                            k_is = len(k_is)
                        else:
                            k_is = k_is['weight'].sum()
                        part[i, t] += np.square(k_is/k_i)
            part[i_at_t, t] = 1 - part[i_at_t, t]
            print(part)
            if decay is not None and t > 0:
                part[i_at_t, t] += decay*part[i_at_t, t-1]
    else:
        for t in np.arange(0, tnet.netshape[1]):
            snapshot = tnet.get_network_when(t=t)
            if tnet.nettype[1] == 'd':
                i_at_t = snapshot['i'].values
            else:
                i_at_t = np.concatenate(
                    [snapshot['i'].values, snapshot['j'].values])
            i_at_t = np.unique(i_at_t).tolist()
            i_at_t = list(map(int, i_at_t))
            for i in i_at_t:
                for tc in np.arange(0, tnet.netshape[1]):
                    C = communities[:, tc]
                    # Calculate degree of node
                    if tnet.nettype[1] == 'd':
                        df = tnet.get_network_when(i=i, t=t)
                        j_at_t = df['j'].values
                        if tnet.nettype[0] == 'w':
                            k_i = df['weight'].sum()
                        elif tnet.nettype[0] == 'b':
                            k_i = len(df)
                    elif tnet.nettype[1] == 'u':
                        df = tnet.get_network_when(ij=i, t=t)
                        j_at_t = np.concatenate(
                            [df['i'].values, df['j'].values])
                        if tnet.nettype == 'wu':
                            k_i = df['weight'].sum()
                        elif tnet.nettype == 'bu':
                            k_i = len(df)
                    j_at_t = list(map(int, j_at_t))

                    for c in np.unique(C[j_at_t]):
                        ci = np.where(C == c)[0].tolist()
                        k_is = tnet.get_network_when(i=i, j=ci, t=t)
                        if tnet.nettype[1] == 'u' and tnet.sparse:
                            k_is2 = tnet.get_network_when(j=i, i=ci, t=t)
                            k_is = pd.concat([k_is, k_is2])
                        if tnet.nettype[0] == 'b':
                            k_is = len(k_is)
                        else:
                            k_is = k_is['weight'].sum()
                        part[i, t] += np.square(k_is/k_i)
                part[i, t] = part[i, t] / tnet.netshape[1]
            part[i_at_t, t] = 1 - part[i_at_t, t]
            if decay is not None and t > 0:
                part[i_at_t, t] += decay*part[i_at_t, t-1]

    # Set any division by 0 to 0
    part[np.isnan(part) == 1] = 0

    return part

