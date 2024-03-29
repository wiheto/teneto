
# import nilearn
import nilearn
import pandas as pd
from ..neuroimagingtools import load_tabular_file

def remove_confounds(timeseries, confounds, confound_selection=None, confound_regex=False, regex_print_selected=False, clean_params=None):
    """
    Removes specified confounds using nilearn.signal.clean

    Parameters
    ----------
    timeseries : array or dataframe
        input timeseries with dimensions: (node,time)
    confounds : array or dataframe
        List of confounds. Expected format is (confound, time).
        If using TenetoBIDS, this does not need to be specified.
    confound_selection : list
        List of confounds. If None, all confounds are removed
    confund_regex : bool
        If True, confound_selection can contain regex expressions
    regex_print_selected : bool
        If True, print the selected confounds after regex, this ensures correct columns are chosen
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
    
    final_confound_selection = []
    if confound_selection is not None:
                    
        if confound_regex:
            for c in confound_selection:
                final_confound_selection += list(confounds.filter(regex=c).columns)
            if regex_print_selected:
                print(final_confound_selection)
        else:
            # Check that confoudns rare in columns
            for c in confound_selection:
                if c not in confounds.columns:
                    raise ValueError('Confound: ' + str(c) +
                                 ' is not in confounds dataframe')
            final_confound_selection = confound_selection
        confounds = confounds[final_confound_selection]

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
