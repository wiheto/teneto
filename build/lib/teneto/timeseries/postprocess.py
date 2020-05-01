"""File contains functions for postprocessing derivation of connectivity estimates"""

import numpy as np
import scipy as sp
from ..utils import set_diagonal


def postpro_fisher(data, report=None):
    """
    Performs fisher transform on everything in data.

    If report variable is passed, this is added to the report.
    """
    if not report:
        report = {}
    # Due to rounding errors
    data[data < -0.99999999999999] = -1
    data[data > 0.99999999999999] = 1
    fisher_data = 0.5 * np.log((1 + data) / (1 - data))
    report['fisher'] = {}
    report['fisher']['performed'] = 'yes'
    #report['fisher']['diagonal'] = 'zeroed'
    return fisher_data, report


def postpro_boxcox(data, report=None):
    """
    Performs box cox transform on everything in data.

    If report variable is passed, this is added to the report.
    """
    if not report:
        report = {}
    # Note the min value of all time series will now be at least 1.
    mindata = 1 - np.nanmin(data)
    data = data + mindata
    ind = np.triu_indices(data.shape[0], k=1)

    boxcox_list = np.array([sp.stats.boxcox(np.squeeze(
        data[ind[0][n], ind[1][n], :])) for n in range(0, len(ind[0]))])

    boxcox_data = np.zeros(data.shape)
    boxcox_data[ind[0], ind[1], :] = np.vstack(boxcox_list[:, 0])
    boxcox_data[ind[1], ind[0], :] = np.vstack(boxcox_list[:, 0])

    bccheck = np.array(np.transpose(boxcox_data, [2, 0, 1]))
    bccheck = (bccheck - bccheck.mean(axis=0)) / bccheck.std(axis=0)
    bccheck = np.squeeze(np.mean(bccheck, axis=0))
    np.fill_diagonal(bccheck, 0)

    report['boxcox'] = {}
    report['boxcox']['performed'] = 'yes'
    report['boxcox']['lambda'] = [
        tuple([ind[0][n], ind[1][n], boxcox_list[n, -1]]) for n in range(0, len(ind[0]))]
    report['boxcox']['shift'] = mindata
    report['boxcox']['shited_to'] = 1

    if np.sum(np.isnan(bccheck)) > 0:
        report['boxcox'] = {}
        report['boxcox']['performed'] = 'FAILED'
        report['boxcox']['failure_reason'] = (
            'Box cox transform is returning edges with uniform values through time. '
            'This is probabaly due to one or more outliers or a very skewed distribution. '
            'Have you corrected for sources of noise (e.g. movement)? '
            'If yes, some time-series might need additional transforms to approximate to Gaussian.'
        )
        report['boxcox']['failure_consequence'] = (
            'Box cox transform was skipped from the postprocess pipeline.'
        )
        boxcox_data = data - mindata
        error_msg = ('TENETO WARNING: Box Cox transform problem. \n'
                     'Box Cox transform not performed. \n'
                     'See report for more details.')
        print(error_msg)

    return boxcox_data, report


def postpro_standardize(data, report=None):
    """
    Standardizes everything in data (along axis -1).

    If report variable is passed, this is added to the report.
    """
    if not report:
        report = {}
    # First make dim 1 = time.
    data = np.transpose(data, [2, 0, 1])
    standardized_data = (data - data.mean(axis=0)) / data.std(axis=0)
    standardized_data = np.transpose(standardized_data, [1, 2, 0])
    report['standardize'] = {}
    report['standardize']['performed'] = 'yes'
    report['standardize']['method'] = 'Z-score'
    # The above makes self connections to nan, set to 1.
    data = set_diagonal(data, 1)
    return standardized_data, report


def postpro_pipeline(data, pipeline, report=None):
    """
    Function to call multiple postprocessing steps.

    Parameters
    -----------

    data : array
        pearson correlation values in temporal matrix form (node,node,time)
    pipeline : list or str
        (if string, each steps seperated by + sign).

            :options: 'fisher','boxcox','standardize'

        Each of the above 3 can be specified. If fisher is used, it must be before boxcox.
        If standardize is used it must be after boxcox and fisher.

    report : bool
        If true, appended to report.

    Returns
    -------

    postpro_data : array
        postprocessed data
    postprocessing_info : dict
        Information about postprocessing

    """

    postpro_functions = {
        'fisher': postpro_fisher,
        'boxcox': postpro_boxcox,
        'standardize': postpro_standardize
    }

    if not report:
        report = {}

    if isinstance(pipeline, str):
        pipeline = pipeline.split('+')

    report['postprocess'] = []
    for postpro_step in pipeline:
        report['postprocess'].append(postpro_step)
        postpro_data, report = postpro_functions[postpro_step](data, report)
    return postpro_data, report
