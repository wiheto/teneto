
import nilearn
from ..neuroimagingtools import load_tabular_file
import pandas as pd


def remove_confounds(timeseries, confounds, confound_selection=None, clean_params=None):
    """
    Removes specified confounds using nilearn.signal.clean

    Parameters
    ----------
    timeseries : array or dataframe
        input timeseries with dimensions: (node,time)
    confounds : array or dataframe
        List of confounds. Expected format is (confound, time).
        If using TenetoBIDS, then the input_params can be bids or bids_<pipeline_name>
    confound_selection : list
        List of confounds. If None, all confoudns are removed
    clean_params : dict
        Dictionary of kawgs to pass to nilearn.signal.clean

    Returns
    -------
    Says all TenetBIDS.get_selected_files with confounds removed with _rmconfounds at the end.

    Note
    ----
    There may be some issues regarding loading non-cleaned data through the TenetoBIDS functions instead of the cleaned data. This depeneds on when you clean the data.
    """

    index = None
    if isinstance(timeseries, pd.DataFrame):
        index = timeseries.index
        timeseries = timeseries.values

    if clean_params is None:
        clean_params = {}

    if isinstance(confounds, str):
        confounds = load_tabular_file(confounds)
    if confound_selection is not None:
        for c in confound_selection:
            if c not in confounds.columns:
                raise ValueError('Confound: ' + str(c) +
                                 ' is not in confounds dataframe')
        confounds = confounds[confound_selection]

    warningtxt = ''
    if confounds.isnull().any().any():
        # Not sure what is the best way to deal with this.
        warningtxt = 'Some confounds contain n/a.\n Setting these values to median of confound.'
        print('WARNING: ' + warningtxt)
        confounds = confounds.fillna(confounds.median())

    if isinstance(confounds, pd.DataFrame):
        confounds = confounds.values

    # nilearn works with time,node data
    timeseries = timeseries.transpose()
    cleaned_timeseries = nilearn.signal.clean(
        timeseries, confounds=confounds, **clean_params)
    cleaned_timeseries = cleaned_timeseries.transpose()
    cleaned_timeseries = pd.DataFrame(cleaned_timeseries)
    if index is not None:
        cleaned_timeseries.index = index
    return cleaned_timeseries
