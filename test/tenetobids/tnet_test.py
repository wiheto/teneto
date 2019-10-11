import teneto
import numpy as np
from PyQt5 import QtCore
import json
import os
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads, True)



def test_tnet_derive():
    # load parc file with data
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.load_data('parcellation')
    tnet.set_confound_pipeline('fmriprep')
    # Turn the confound_corr_report to True once matplotlib works withconcurrent
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time'},
                                update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    parcdata = tnet.parcellation_data_[0]
    parcdata.drop('0', axis=1, inplace=True)
    R_jc = parcdata.transpose().corr().values[0, 1] * -1
    jc = float(tnet.tvc_data_[0][(tnet.tvc_data_[0]['i'] == 0) & (
        tnet.tvc_data_[0]['j'] == 1) & (tnet.tvc_data_[0]['t'] == 0)]['weight'])
    if not np.round(R_jc, 12) == np.round(jc, 12):
        raise AssertionError()
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)


def test_make_fc_and_tvc():
    # Load parc data, make FC JC method
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.load_data('parcellation')
    r = tnet.make_functional_connectivity(returngroup=True)[0, 1]
    fc_files = tnet.get_selected_files(pipeline='functionalconnectivity')
    if not '_conn.tsv' in fc_files[0]:
        raise AssertionError()
    if not len(fc_files) == 1:
        raise AssertionError()
    R = tnet.parcellation_data_[0].transpose().corr().values[0, 1]
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time',
                                 'postpro': 'standardize'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JC = tnet.tvc_data_[0].iloc[0].values[-1]
    # Load parc data, make FC JC method with FC dual weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time', 'weight-mean': 'from-subject-fc',
                                 'weight-var': 'from-subject-fc'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JCw = tnet.tvc_data_[0].iloc[0].values[-1]
    # Load parc data, make FC JC method with FC mean weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time',
                                 'weight-mean': 'from-subject-fc'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JCm = tnet.tvc_data_[0].iloc[0].values[-1]
    # Load parc data, make FC JC method with FC variance weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time',
                                 'weight-var': 'from-subject-fc'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JCv = tnet.tvc_data_[0].iloc[0].values[-1]
    if not np.round(JCw, 15) == np.round((JC*r)+R, 15):
        raise AssertionError()
    if not np.round(JCv, 15) == np.round((JC*r), 15):
        raise AssertionError()
    if not np.round(JCm, 15) == np.round((JC)+R, 15):
        raise AssertionError()


def test_communitydetection():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='tvc', bids_suffix='tvcconn', bids_tags={'sub': '001', 'task': 'b', 'run': '01'}, raw_data_exists=False)
    community_detection_params = {'resolution': 1,
                                  'intersliceweight': 0}
    tnet.communitydetection(community_detection_params, 'temporal')
    # Compensating for data not being in a versioen directory
    tnet.set_pipeline('teneto_' + teneto.__version__)
    tnet.load_data('communities')
    # not creating folder in travis (commenting out for now)
    # C =  tnet.communities_data_[0].values
    # if not C[0, 0] == C[1, 0] == C[2, 0]:
    #     raise AssertionError()
    # if not C[3, 0] == C[4, 0] == C[5, 0]:
    #     raise AssertionError()
    # if not C[0, 2] == C[1, 2] == C[2, 2] == C[3, 2]:
    #     raise AssertionError()
    # if not C[4, 2] == C[5, 2]:
    #     raise AssertionError()
    # if not C[3, 0] != C[0, 0]:
    #     raise AssertionError()
    # if not C[4, 2] != C[0, 2]:
    #     raise AssertionError()




# def test_timelockednetworkmeasure():
#     # calculate and load a network measure
#     tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto_' + teneto.__version__,
#                              pipeline_subdir='tvc', bids_suffix='tvc', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
#     tnet.networkmeasures('volatility', {'calc': 'time'}, save_tag='time')
#     tnet.load_data('temporalnetwork', measure='volatility')
#     vol = tnet.temporalnetwork_data_['volatility'][0].values
#     tnet.make_timelocked_events('volatility', 'testevents', [
#                                 1], [-1, 1], tag='time')
#     tnet.load_timelocked_data('volatility')
#     if not np.all(np.squeeze(tnet.timelocked_data_) == np.squeeze(tnet.networkmeasure_[0, 0:3]))
#:
# raise AssertionError()


def test_tnet_derive_with_removeconfounds():
    # load parc file with data
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    # Set gthe confound pipeline in fmriprep
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    if not 'confound1' in alt:
        raise AssertionError()
    if not 'confound2' in alt:
        raise AssertionError()
    # Set the confounds
    tnet.set_confounds('confound1')
    # Remove confounds
    tnet.removeconfounds(transpose=True)
    f = tnet.get_selected_files()[0]
    f = f.replace('.tsv', '.json')
    with open(f) as fs:
        sidecar = json.load(fs)

    # Removing below tests due to errors caused by concurrent images.
    #tnet.derive_temporalnetwork({'method': 'jackknife'})
    # Make sure report directory exists
    # if not os.path.exists(teneto.__path__[0] + '/data/testdata/dummybids/derivatives/teneto_' + teneto.__version__ + '/sub-001/func/tvc/report'):
    #raise AssertionError()


def test_tnet_scrubbing():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '>1', replace_with='nan')


def test_tnet_scrubbing_and_spline():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.load_data('parcellation')
    dat_orig = np.squeeze(tnet.parcellation_data_[0].values)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '>1', replace_with='cubicspline')
    tnet.load_data('parcellation')
    dat_scrub = tnet.parcellation_data_[0].values
    # Make sure there is a difference
    if not np.sum(dat_scrub != dat_orig):
        raise AssertionError()


def test_tnet_set_bad_files():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.load_data('parcellation')
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_file('confound2', '>0')



def test_tnet_make_parcellation():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='fmriprep',
                             bids_suffix='bold', bids_tags={'sub': '001', 'task': 'a', 'run': '01', 'desc': 'preproc'}, raw_data_exists=False)
    tnet.make_parcellation(atlas='Schaefer2018',
                           atlas_desc='100Parcels17Networks')
    tnet.load_data('parcellation')
    # Hard coded facts about dummy data
    if not tnet.parcellation_data_[0].shape == (100, 10):
        raise AssertionError()


def test_tnet_checksidecar():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='fmriprep',
                             bids_suffix='bold', bids_tags={'sub': '001', 'task': 'a', 'run': '01', 'desc': 'preproc'}, raw_data_exists=False)
    tnet.make_parcellation(atlas='Schaefer2018',
                           atlas_desc='100Parcels17Networks')
    tnet.load_data('parcellation')
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '<=0', replace_with='nan')
    with open(teneto.__path__[0] + '/data/testdata/dummybids/derivatives/teneto_' + teneto.__version__ + '/sub-001/func/parcellation/sub-001_task-a_run-01_desc-preproc_roi.json') as fs:
        sidecar = json.load(fs)
    # Check both steps are in sidecar
    if not 'parcellation' in sidecar.keys():
        raise AssertionError()
    if not 'scrubbed_timepoints' in sidecar.keys():
        raise AssertionError()


def test_export_history():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='fmriprep',
                             bids_suffix='preproc', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    export_path = teneto.__path__[0] + '/data/testdata/dummybids/'
    tnet.export_history(export_path)
    if not os.path.exists(export_path + 'requirements.txt'):
        raise AssertionError()
    if not os.path.exists(export_path + 'TenetoBIDShistory.py'):
        raise AssertionError()
    if not os.path.exists(export_path + 'tenetoinfo.json'):
        raise AssertionError()


def test_tnet_scrubbing_and_exclusion_options():
    # <=
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '<=0', replace_with='nan')
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_file('confound2', '<=1')
    # <
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '<0', replace_with='nan')
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_file('confound2', '<1')
    # >=
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound2', '>=2', replace_with='nan')
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_file('confound2', '>=1')


def test_savesnapshot():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': '01'}, raw_data_exists=False)
    tnet.save_tenetobids_snapshot(teneto.__path__[0])
    with open(teneto.__path__[0] + '/TenetoBIDS_snapshot.json') as f:
        params = json.load(f)
    tnet2 = teneto.TenetoBIDS(**params)
    for n in tnet2.__dict__:
        if tnet.__dict__[n] != tnet2.__dict__[n]:
            raise AssertionError()
    if tnet2.__dict__.keys() != tnet.__dict__.keys():
        raise AssertionError()

def test_networkmeasure():
    # calculate and load a network measure
    bids_path = teneto.__path__[0] + '/data/testdata/dummybids/'
    pipeline = 'teneto_' + teneto.__version__
    tags = {'sub': '001', 'task': 'a', 'run': '01'}
    tnet = teneto.TenetoBIDS(bids_path, pipeline=pipeline, pipeline_subdir='tvc',
                             bids_suffix='tvcconn', bids_tags=tags,
                             raw_data_exists=False)
    tnet.networkmeasures('volatility', {'calc': 'time'}, tag='time')
    tnet.load_data('temporalnetwork', measure='volatility', tag='time')
    vol = tnet.temporalnetwork_data_['volatility']
