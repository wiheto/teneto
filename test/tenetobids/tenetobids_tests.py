from teneto import TenetoBIDS
from teneto import __path__ as tenetopath
import numpy as np
datdir = tenetopath[0] + '/data/testdata/dummybids/'
tnet = TenetoBIDS(datdir, selected_pipeline='fmriprep', bids_filters={
                     'subject': '001', 'run': 1, 'task': 'a'}, exist_ok=True)
tnet.run('make_parcellation', {'atlas': 'Schaefer2018',
                               'atlas_desc': '100Parcels7Networks',
                               'parc_params': {'detrend': True}})
tnet.run('remove_confounds', {'confound_selection': ['confound1']})
tnet.run('derive_temporalnetwork', {'params': {
         'method': 'jackknife', 'postpro': 'standardize'}})
tnet.run('binarize', {'threshold_type': 'percent', 'threshold_level': 0.1})
tnet.run('volatility', {})
vol = tnet.load_data()
# Hard code truth 
if np.round(np.squeeze(vol['sub-001_run-1_task-a_vol.tsv'].values),5) != 0.10527:
    raise AssertionError()