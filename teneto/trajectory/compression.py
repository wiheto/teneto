import teneto
import numpy as np


def rdp(datin, delta=1, report=10, quiet=True):
    """
    """
    # This needs to be added to utils for trajectory detection
    # T will be for trajectory or timeseries data, (roi x time).
    datin, datinfo = teneto.utils.process_input(datin, ['C', 'G', 'TO', 'T'])
    # if network, then make to roi,time shape. If T, nothing needs to be done.
    if len(datin.shape) == 3:
        ind = np.triu_indices(datin.shape[0], k=1)
        datin = datin[ind[0], ind[1], :]
        index_out = np.array([ind[0], ind[1]]).transpose()
    else:
        index_out = np.arange(0, datin.shape[0])

    # Interprets data to be rois x time.
    datin = np.array(datin, ndmin=2)
    s = 0
    e = datin.shape[-1]

    # Create straight line between start and end point
    trajectory = teneto.utils.create_traj_ranges(
        datin[:, s], datin[:, e-1], e-s)

    # Create lists of trajectories
    trajectory_points = [np.array([s, e-1]) for n in np.arange(datin.shape[0])]
    # Preset some outputs
    reduction = np.zeros(len(trajectory_points))
    error = np.zeros(len(trajectory_points))
    amp_weighted_error = np.zeros(len(trajectory_points))
    # Stopping condition and round preset
    stop_cond = 0
    round_count = 0
    while stop_cond < datin.shape[0]:
        traj_data_diff = np.abs(datin-trajectory)
        ind = np.argmax(traj_data_diff, axis=1)

        for i, ix in enumerate(ind):
            ind_bool = traj_data_diff[i, ix] > delta
            if ind_bool:
                # Get trajectory breaking point indicies
                traj_ind = np.searchsorted(
                    trajectory_points[i], ix, side='left')
                traj_start = trajectory_points[i][traj_ind-1]
                traj_end = trajectory_points[i][traj_ind]
                # Make new trajectories
                r1 = np.arange(traj_start, ix+1)
                trajectory[i, r1] = teneto.utils.create_traj_ranges(
                    trajectory[i, traj_start], datin[i, ix], len(r1))
                r2 = np.arange(ix, traj_end+1)
                trajectory[i, r2] = teneto.utils.create_traj_ranges(
                    datin[i, ix], trajectory[i, traj_end], len(r2))
                # Add new point
                trajectory_points[i] = np.insert(
                    trajectory_points[i], traj_ind, ix)
            else:
                stop_cond += 1

        round_count += 1
        if np.remainder(round_count, report) == 0 and not quiet:
            print('After round: ' + str(round_count) + ', ' +
                  str(np.round((stop_cond/datin.shape[0])*100, 2)) + '% completed data reduction')

    for i in np.arange(datin.shape[0]):
        reduction[i] = float(1-len(trajectory_points[i])/datin.shape[1])
        error[i] = float(np.mean(traj_data_diff[i]))
        amp_weighted_error[i] = float(
            np.mean(np.abs(datin[i, :])*traj_data_diff[i]))

    # Construct ouput dictionary
    traj = datinfo
    traj['trajectory'] = trajectory
    traj['trajectory_points'] = trajectory_points
    traj['reduction'] = reduction
    traj['error'] = error
    traj['weighted_error'] = amp_weighted_error
    traj['nettype'] = 't'
    traj['index'] = index_out
    traj.pop('inputtype')

    return traj
