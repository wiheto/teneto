import numpy as np
import pandas as pd
import os
import json
from scipy.interpolate import interp1d

def drop_bids_suffix(fname):
    """
    Given a filename sub-01_run-01_preproc.nii.gz, it will return ['sub-01_run-01', '.nii.gz']

    Parameters
    ----------

    fname : str
        BIDS filename with suffice. Directories should not be included.

    Returns
    -------
    fname_head : str
        BIDS filename with
    fileformat : str
        The file format (text after suffix)

    Note
    ------
    This assumes that there are no periods in the filename
    """
    if '/' in fname:
        split = fname.split('/')
        dirnames = '/'.join(split[:-1]) + '/'
        fname = split[-1]
    else:
        dirnames = ''
    tags = [tag for tag in fname.split('_') if '-' in tag]
    fname_head = '_'.join(tags)
    fileformat = '.' + '.'.join(fname.split('.')[1:])
    return dirnames + fname_head, fileformat


def load_tabular_file(fname, header=True, index_col=True):
    """
    Given a file name loads as a pandas data frame

    Parameters
    ----------
    fname : str
        file name and path. Must be tsv.
    header : bool (default True)
        if there is a header in the tsv file, true will use first row in file.
    index_col : bool (default None)
        if there is an index column in the csv or tsv file, true will use first row in file.

    Returns
    -------
    df : pandas
        The loaded file
    info : pandas, if return_meta=True
        Meta infomration in json file (if specified)
    """
    if index_col:
        index_col = 0
    else:
        index_col = None
    if header:
        header = 0
    else:
        header = None

    df = pd.read_csv(fname, header=header, index_col=index_col, sep='\t')
    return df


def get_sidecar(fname, allowedfileformats='default'):
    """Loads sidecar or creates one"""
    if allowedfileformats == 'default':
        allowedfileformats = ['.tsv', '.nii.gz']
    for f in allowedfileformats:
        fname = fname.split(f)[0]
    fname += '.json'
    if os.path.exists(fname):
        with open(fname) as fs:
            sidecar = json.load(fs)
    else:
        sidecar = {}
    if 'BadFile' not in sidecar:
        sidecar['BadFile'] = False
    return sidecar


def process_exclusion_criteria(exclusion_criteria):
    r"""
    Parses an exclusion critera string to get the function and threshold.

    Parameters
    ----------
        exclusion_criteria : list
            list of strings where each string is of the format [relation][threshold]. E.g. \'<0.5\' or \'>=1\'

    Returns
    -------
        relfun : func
            numpy functions for the exclusion criteria
        threshold : float
            floats for threshold for each relfun


    """
    if exclusion_criteria[1] == '=':
        splitat = 2
    else:
        splitat = 1
    exfuncs = {'>=': np.greater_equal,
               '<=': np.less_equal,
               '>': np.greater,
               '<': np.less}
    relfun = exfuncs[exclusion_criteria[:splitat]]
    threshold = float(exclusion_criteria[splitat:])
    return relfun, threshold


def exclude_runs(sidecar, confounds, confound_name, exclusion_criteria, confound_stat='mean'):
    """
    Excludes subjects given a certain exclusion criteria.

    Parameters
    ----------
        confounds : dataframe
            dataframe of confounds
        confound_name : str
            Confound name from confound files that is to be used
        exclusion_criteria  : str
            An exclusion_criteria should be expressed as a string.
            It starts with >,<,>= or <= then the numerical threshold.
            Eg. '>0.2' will entail every subject with the avg greater than 0.2 of confound will be rejected.
        confound_stat : str or list
            Can be median, mean, std.
            How the confound data is aggregated (so if there is a meaasure per time-point, this is averaged over all time points.
            If multiple confounds specified, this has to be a list.).
    Returns
    --------
        calls TenetoBIDS.set_bad_files with the files meeting the exclusion criteria.
    """
    # Checks could be made regarding confound number
    if confound_name not in confounds:
        raise ValueError('Confound_name not found')
    relex, crit = process_exclusion_criteria(exclusion_criteria)
    found_bad_subject = False
    relfuncs = {'median': np.nanmedian,
                'mean': np.nanmean,
                'std': np.std}
    if relex(relfuncs[confound_stat](confounds[confound_name]), crit):
        found_bad_subject = True
    # If file is confound.
    if found_bad_subject:
        sidecar['BadFile'] = True
        sidecar['file_exclusion'] = {}
        sidecar['file_exclusion']['confound'] = confound_name
        sidecar['file_exclusion']['threshold'] = exclusion_criteria
        sidecar['file_exclusion']['stat'] = confound_stat
    return sidecar


def censor_timepoints(timeseries, sidecar, confounds, confound_name, exclusion_criteria, replace_with, tol=1):
    """
    Excludes subjects given a certain exclusion criteria.

    Does not work on nifti files, only tsv. Assumes data is node,time.
    Assumes the time-point column names are integers.

    Parameters
    ----------
        timeseries : dataframe
            dataframe of time series
        sidecar : dict
            json sidecar in dict format
        confounds : dataframe
            dataframe of confounds
        confound_name : str
            string of confound name from confound files.
        exclusion_criteria  : str or list
            for each confound, an exclusion_criteria should be expressed as a string.
            It starts with >,<,>= or <= then the numerical threshold.
            Ex. '>0.2' will entail every subject with the avg greater than 0.2 of confound will be rejected.
        replace_with : str
            Can be 'nan' (bad values become nans) or 'cubicspline' (bad values are interpolated).
            If bad value occurs at 0 or -1 index, then these values are kept and no interpolation occurs.
        tol : float
            Tolerance of exlcuded time-points allowed before being set a BadFile in sidecar.
            If 0.25, then 25% of time-points can be marked censored/replaced before being a BadFile.

    Returns
    ------
        Loads the TenetoBIDS.selected_files and replaces any instances of confound meeting the exclusion_criteria with replace_with.
    """
    relex, crit = process_exclusion_criteria(exclusion_criteria)
    ci = confounds[confound_name]
    bad_timepoints = list(ci[relex(ci, crit)].index)
    bad_timepoints = list(map(str, bad_timepoints))
    timeseries[bad_timepoints] = np.nan
    if replace_with == 'cubicspline' and len(bad_timepoints) > 0:
        good_timepoints = sorted(
            np.array(list(map(int, set(timeseries.columns).difference(bad_timepoints)))))
        bad_timepoints = np.array(list(map(int, bad_timepoints)))
        ts_index = timeseries.index
        timeseries = timeseries.values
        bt_interp = bad_timepoints[bad_timepoints > np.min(good_timepoints)]
        for n in range(timeseries.shape[0]):
            interp = interp1d(
                good_timepoints, timeseries[n, good_timepoints], kind='cubic')
            timeseries[n, bt_interp] = interp(bt_interp)
        timeseries = pd.DataFrame(timeseries, index=ts_index)
        bad_timepoints = list(map(str, bad_timepoints))

    if len(bad_timepoints) / timeseries.shape[1] > tol:
        sidecar['BadFile'] = True
        sidecar['file_exclusion'] = {}
        sidecar['file_exclusion']['confound'] = confound_name
        sidecar['file_exclusion']['threshold'] = exclusion_criteria
        sidecar['file_exclusion']['tolerance_level'] = float(tol)
        sidecar['file_exclusion']['reason'] = 'Time-points exceded tolerance level'

    # update sidecar
    sidecar['censored_timepoints'] = {}
    sidecar['censored_timepoints']['description'] = 'Censors timepoints where the confounds met exclusion crtiera a certain time-points.\
        Censored time-points are replaced with replacement value (nans or cubic spline).'
    sidecar['censored_timepoints'][confound_name] = {}
    sidecar['censored_timepoints'][confound_name]['threshold'] = exclusion_criteria
    sidecar['censored_timepoints'][confound_name]['replacement'] = replace_with
    sidecar['censored_timepoints'][confound_name]['badpoint_number'] = len(
        bad_timepoints)
    sidecar['censored_timepoints'][confound_name]['badpoints'] = ','.join(
        bad_timepoints)
    sidecar['censored_timepoints'][confound_name]['badpoint_ratio'] = float(
        len(bad_timepoints) / timeseries.shape[1])
    sidecar['censored_timepoints'][confound_name]['file_exclusion_when_badpoint_ratio'] = float(
        tol)

    return timeseries, sidecar
