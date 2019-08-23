import numpy as np
import pandas as pd
import os
import json


def make_directories(path):
    """
    """
    # Updated function to this and will eventuall merge remove function if this does not raise error when in parallel
    os.makedirs(path, exist_ok=True)
    # # Error can occur with os.makedirs when parallel so here a try/error is added to fix that.
    # if not os.path.exists(path):
    #     try:
    #         os.makedirs(path, exist_ok=True)
    #     except:
    #         time.sleep(5)


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


def get_bids_tag(filename, tag):
    """
    """
    outdict = {}
    filename, _ = drop_bids_suffix(filename)
    if isinstance(tag, str):
        if tag == 'all':
            for t in filename.split('_'):
                tag = t.split('-')
                if len(tag) == 2:
                    outdict[tag[0]] = tag[1]
            tag = 'all'
        else:
            tag = [tag]
    if isinstance(tag, list):
        if '/' in filename:
            filename = filename.split('/')[-1]
        for t in tag:
            if t in filename:
                outdict[t] = filename.split(t + '-')[1].split('_')[0]
    if 'run' in outdict:
        outdict['run'] = str(int(outdict['run']))
    return outdict


def load_tabular_file(fname, return_meta=False, header=True, index_col=True):
    """
    Given a file name loads as a pandas data frame

    Parameters
    ----------
    fname : str
        file name and path. Must be tsv.
    return_meta :

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

    if return_meta:
        json_fname = fname.replace('tsv', 'json')
        meta = pd.read_json(json_fname)
        return df, meta
    else:
        return df


def get_sidecar(fname, allowedfileformats='default'):
    """
    Loads sidecar or creates one
    """
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


def confound_matching(files, confound_files):
    """
    """
    files_out = []
    confounds_out = []
    files_taglist = []
    confound_files_taglist = []
    for f in files:
        tags = get_bids_tag(f, ['sub', 'ses', 'run', 'task'])
        files_taglist.append(tags.values())
    for f in confound_files:
        tags = get_bids_tag(f, ['sub', 'ses', 'run', 'task'])
        confound_files_taglist.append(tags.values())

    for i, t in enumerate(files_taglist):
        j = [j for j, ct in enumerate(
            confound_files_taglist) if list(t) == list(ct)]
        if len(j) > 1:
            raise ValueError(
                'File/confound matching error (more than one confound file identified)')
        if len(j) == 0:
            raise ValueError(
                'File/confound matching error (no confound file found)')
        files_out.append(files[i])
        confounds_out.append(confound_files[j[0]])
    return files_out, confounds_out


def process_exclusion_criteria(exclusion_criteria):
    """
    Parses an exclusion critera string to get the function and threshold.

    Parameters
    ----------
        exclusion_criteria : list
            list of strings where each string is of the format [relation][threshold]. E.g. \'<0.5\' or \'>=1\'

    Returns
    -------
        relfun : list
            list of numpy functions for the exclusion criteria
        threshold : list
            list of floats for threshold for each relfun


    """
    relfun = []
    threshold = []
    for ec in exclusion_criteria:
        if ec[0:2] == '>=':
            relfun.append(np.greater_equal)
            threshold.append(float(ec[2:]))
        elif ec[0:2] == '<=':
            relfun.append(np.less_equal)
            threshold.append(float(ec[2:]))
        elif ec[0] == '>':
            relfun.append(np.greater)
            threshold.append(float(ec[1:]))
        elif ec[0] == '<':
            relfun.append(np.less)
            threshold.append(float(ec[1:]))
        else:
            raise ValueError('exclusion crieria must being with >,<,>= or <=')
    return relfun, threshold
