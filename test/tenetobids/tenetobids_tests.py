from teneto import TenetoBIDS

datdir = '/home/william/work/teneto/teneto/data/testdata/dummybids/'
tnet = TenetoBIDS(datdir, selected_pipeline='fmriprep', bids_filters={
                     'subject': '001', 'run': 1, 'task': 'a'}, overwrite=True)
tnet.run('make_parcellation', {'atlas': 'Schaefer2018',
                               'atlas_desc': '100Parcels7Networks',
                               'parc_params': {'detrend': True}})
tnet.run('remove_confounds', {'confound_selection': ['confound1']})
tnet.run('derive_temporalnetwork', {'params': {
         'method': 'jackknife', 'postpro': 'standardize'}})
tnet.run('binarize', {'threshold_type': 'percent', 'threshold_level': 0.1})
tnet.run('volatility', {})
