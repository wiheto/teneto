"""TCTC is a temporal community detection method on nodal time series using trajectories."""
import numpy as np
import networkx as nx
import pandas as pd
from teneto.timeseries import derive_temporalnetwork


def partition_inference(tctc_mat, comp, tau, sigma, kappa):
    r"""
    Takes tctc trajectory matrix and returns dataframe where all multi-label communities are listed

    Can take a little bit of time with large datasets and optimizaiton could remove some for loops.
    """
    communityinfo = {}
    communityinfo['community'] = []
    communityinfo['start'] = np.empty(0)
    communityinfo['end'] = np.empty(0)
    communityinfo['size'] = np.empty(0)
    for i, tcomp in enumerate(comp):
        # This can go in parallel loop
        if len(tcomp) > 0:
            for traj in tcomp:
                # Check it does not already exist.
                ignore = 0
                preexisting = 0
                if i != 0:
                    cutoff = i-1-kappa
                    if cutoff < 0:
                        cutoff = 0

                    if np.any(np.sum(np.sum(tctc_mat[traj, :, cutoff:i][:, traj], axis=0), axis=0) == np.power(len(traj), 2)):
                        # Make sure that a small trajectory could exist
                        for checknode in np.where(communityinfo['end'] >= cutoff)[0]:
                            if traj == communityinfo['community'][checknode]:
                                ignore = 1
                        if ignore == 0:
                            for checknode in np.where(communityinfo['end'] >= cutoff)[0]:
                                if set(communityinfo['community'][checknode]).issuperset(traj):
                                    preexisting = 1
                if ignore == 0:
                    # Check how long it continues
                    # For efficiency, increase in blocks
                    approxmaxlength = tau*2
                    a = np.sum(
                        np.sum(tctc_mat[traj, :, i:i+approxmaxlength][:, traj], axis=0), axis=0)
                    if len(traj)*len(traj)*approxmaxlength == a.sum():
                        ok = 0
                        ii = 1
                        while ok == 0:
                            b = np.sum(np.sum(
                                tctc_mat[traj, :, i+(approxmaxlength*ii):i+(approxmaxlength*(ii+1))][:, traj], axis=0), axis=0)
                            a = np.append(a, b)
                            if len(traj)*len(traj)*approxmaxlength != b.sum():
                                ok = 1
                            if i+(approxmaxlength*(ii+1)) > tctc_mat.shape[-1]:
                                ok = 1
                            ii += 1
                    a = np.where(a == np.power(len(traj), 2))[0]
                    # Add an additional value that is false in case end of time series
                    if len(a) == 1:
                        stopind = i + 1
                    else:
                        a = np.append(a, a.max()+kappa+2)
                        # Find the stop index (if stopind = 4 and start = 0, then tctc_mat[:,:,start:stopind]==1)
                        stopind = i + np.split(a, np.where(
                            np.diff(a) > kappa+1)[0]+1)[0][-1] + 1
                    # Add trajectory to dictionary
                    if ((stopind - i) >= tau or preexisting == 1) and len(traj) >= sigma:
                        communityinfo['community'].append(sorted(traj))
                        communityinfo['start'] = np.append(
                            communityinfo['start'], int(i))
                        communityinfo['end'] = np.append(
                            communityinfo['end'], int(stopind))
                        communityinfo['size'] = np.append(
                            communityinfo['size'], len(traj))

    communityinfo = pd.DataFrame(communityinfo)

    communityinfo['start'] = communityinfo['start'].astype(int)
    communityinfo['end'] = communityinfo['end'].astype(int)
    # First check that there is not already a trajectory that is ongoing
    badrows = []
    for v in communityinfo.iterrows():
        skipselrule = (communityinfo['end'] == v[1]['end'])
        for u in communityinfo[skipselrule].iterrows():
            a = 1
            if u[1]['start'] > v[1]['start'] and sorted(u[1]['community']) == sorted(v[1]['community']):
                badrows.append(u[0])
    communityinfo = communityinfo.drop(badrows)

    # Then see if any subset trajectory can be placed earlier in time.
    for v in communityinfo.iterrows():
        skipselrule = (communityinfo['end'] <= v[1]['start']) & (
            communityinfo['end']+kappa >= v[1]['start'])
        for u in communityinfo[skipselrule].iterrows():
            a = 1
            if set(u[1]['community']).issuperset(v[1]['community']):
                communityinfo.loc[v[0], 'start'] = u[1]['start']

    # It is possible to make the condition below effective_length
    communityinfo['length'] = np.array(
        communityinfo['end']) - np.array(communityinfo['start'])
    communityinfo = communityinfo[communityinfo['length'] >= tau]
    communityinfo = communityinfo[communityinfo['size'] >= sigma]

    # Make sure that the traj is not completely enguled by another
    badrows = []
    if kappa > 0:
        for v in communityinfo.iterrows():
            skipselrule = (communityinfo['end'] == v[1]['end']) & (
                communityinfo['start'] < v[1]['start'])
            for u in communityinfo[skipselrule].iterrows():
                if set(v[1]['community']).issubset(u[1]['community']):
                    badrows.append(v[0])
        communityinfo = communityinfo.drop(badrows)

    return communityinfo


def tctc(data, tau, epsilon, sigma, kappa=0, largedataset=False,
         rule='flock', noise=None, raw_signal='amplitude', output='array',
         tempdir=None, njobs=1, largestonly=False):
    r"""
    Runs TCTC community detection

    Parameters
    ----------
    data : array
        Multiariate series with dimensions: "time, node" that belong to a network.
    tau : int
        tau specifies the minimum number of time-points of each temporal community must last.
    epsilon : float
        epsilon specifies the distance points in a community can be away from each other.
    sigma : int
        sigma specifies the minimum number of nodes that must be in a community.
    kappa : int
        kappa specifies the number of consecutive time-points that can break the distance or size rules.
    largedataset : bool
        If true, runs with HDF5 (beta)
    rule : str
        Can be 'convoy' or 'flock'.
            - flock entials all nodes are max epsilon apart in a communiy.
            - convoy entails that there is at least one node that is epsilon apart.
    noise : array (defauly None)
        Timeseries of dimensions "time, N" where N is the number of noise time series added. Any community that contains this time series is excluded.
    raw_signal : str
        Can be amplitude or phase
    output : str
        Can be array or df or None
    tempdir : str
        Specify where the temporary directory is if largedataset is True
    njobs : int
        number of jobs (not implemented yet)
    largestonly : bool (default False)
        If True only considers largest communities in rule application (should generally be false)

    Returns
    -----------
        tctc : array, df
    """
    # Get distance matrix
    if largedataset:
        raise NotImplementedError(
            'HDF5 implementation for large datasets is not available yet')
    else:
        N_data = data.shape[1]
        if noise is not None:
            if len(noise.shape) == 1:
                noise = np.array(noise, ndmin=2).transpose()
            N_data = data.shape[1]
            data = np.hstack([data, noise])

        N = data.shape[1]
        #T = data.shape[0]

        if raw_signal == 'amplitude':
            d = np.array([np.abs(data[:, n]-data[:, m])
                          for n in range(data.shape[-1]) for m in range(data.shape[-1])])
            d = np.reshape(d, [data.shape[-1], data.shape[-1], data.shape[0]])

        elif raw_signal == 'phase':
            params = {'method': 'ips', 'dimord': 'time,node'}
            # Correct for inversion
            d = 1 - derive_temporalnetwork(data, params)

        # Shape of datin (with any addiitonal 0s or noise added to nodes)
        dat_shape = [int(d.shape[-1]), int(d.shape[0])]
        # Make trajectory matrix 1 where distance critera is kept
        tctc_mat = np.zeros([dat_shape[1], dat_shape[1], dat_shape[0]])
        tctc_mat[:, :, :][d <= epsilon] = 1

        t1 = 1
        t2 = 2
        # The next two rules have to be run iteratively until it converges.
        # i.e. when applying the sigma and tau parameters, if nothing more is pruned, then this is complete
        # There may be a case where running it in this order could through some value that is unwanted due to the skipping mechanic.
        # Doing it in the other order does create possible bad values.
        while t1 != t2:

            t1 = tctc_mat.sum()
            cliques = []
            if tctc_mat.sum() > 0:
                # Run the trajectory clustering rule
                if rule == 'flock':

                    cliques = [list(filter(lambda x: (len(x) >= sigma) and (len(set(x).intersection(np.arange(N_data, N+1))) == 0), nx.find_cliques(
                        nx.Graph(tctc_mat[:, :, t])))) for t in range(tctc_mat.shape[-1])]
                    #cliques = []
                    # with ProcessPoolExecutor(max_workers=njobs) as executor:
                    #    job = {executor.submit(_cluster_flocks,tctc_mat[:,:,t],sigma) for t in range(tctc_mat.shape[-1])}
                    #    for j in as_completed(job):
                    #        cliques.append(j.result()[0])

                elif rule == 'convoy':
                    cliques = [list(map(list, filter(lambda x: (len(x) >= sigma) and
                                                     (len(set(x).intersection(np.arange(N_data, N+1))) == 0), nx.connected_components(
                        nx.Graph(tctc_mat[:, :, t]))))) for t in range(tctc_mat.shape[-1])]

                # Reset the trajectory matrix (since info is now in "cliques").
                # Add the infomation from clique into tctc_mat (i.e sigma is now implemented)
                tctc_mat = np.zeros([dat_shape[1], dat_shape[1], dat_shape[0]])
                # Due to advanced index copy, I've done this with too many forloops
                for t in range(dat_shape[0]):
                    for c in cliques[t]:
                        # Make one of index communitytors a list.
                        cv = [[i] for i in c]
                        tctc_mat[cv, c, t] = 1

            if tctc_mat.sum() > 0:
                # Now impose tau criteria. This is done by flattening and (since tau has been added to the final dimension)
                # Add some padding as this is going to be needed when flattening (ie different lines must have at least tau+kappa spacing between them)
                tctc_mat = np.dstack([np.zeros([dat_shape[1], dat_shape[1], 1]), tctc_mat, np.zeros(
                    [dat_shape[1], dat_shape[1], tau+kappa])])
                # Make to singular communitytor
                tctc_mat_community = np.array(tctc_mat.flatten())
                # Add an extra 0
                tctc_mat_dif = np.append(tctc_mat_community, 0)
                # Use diff. Where there is a 1 trajectory starts, where -1 trajectory ends
                tctc_mat_dif = np.diff(tctc_mat_dif)
                start_ones = np.where(tctc_mat_dif == 1)[0]
                end_ones = np.where(tctc_mat_dif == -1)[0]
                skip_ind = np.where(start_ones[1:]-end_ones[:-1] <= kappa)[0]
                start_ones = np.delete(start_ones, skip_ind+1)
                end_ones = np.delete(end_ones, skip_ind)

                traj_len = end_ones - start_ones
                # whereever traj_len is not long enough, loop through ind+t and make these 0
                ind = start_ones[traj_len >= tau] + 1
                l2 = traj_len[traj_len >= tau]
                # for t in range(tau-1): # this didn't work (but was quicker) because of tau bug
                #    tctc_mat[ind+t] = 0
                # Looping over each valid trajectory instance is slower but the safest was to impose tau restrain and reinserting it.
                tctc_mat = np.zeros(tctc_mat_community.shape)
                for i, irow in enumerate(ind):
                    tctc_mat[irow:irow+l2[i]] = 1
                tctc_mat = tctc_mat.reshape(
                    dat_shape[1], dat_shape[1], dat_shape[0]+kappa+tau+1)
                # remove padding
                tctc_mat = tctc_mat[:, :, 1:dat_shape[0]+1]

            t2 = tctc_mat.sum()

        # remove noise
        tctc_mat = tctc_mat[:N_data, :N_data]
        if output == 'array':
            return tctc_mat

        elif output == 'df':
            if np.sum(tctc_mat) != 0:
                df = partition_inference(
                    tctc_mat, cliques, tau, sigma, kappa)
                return df
            else:
                return []
