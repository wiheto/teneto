import teneto
import numpy as np
import os
from PyQt5 import QtCore
import json
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads, True)


def test_tnet_derive():
    # load parc file with data
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.load_data('parcellation')
    tnet.set_confound_pipeline('fmriprep')
    # Turn the confound_corr_report to True once matplotlib works withconcurrent
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    parcdata = tnet.parcellation_data_[0]
    parcdata.drop('0', axis=1, inplace=True)
    R_jc = parcdata.transpose().corr().values[0,1] * -1
    jc = float(tnet.tvc_data_[0][(tnet.tvc_data_[0]['i'] == 0) & (tnet.tvc_data_[0]['j'] == 1) & (tnet.tvc_data_[0]['t'] == 0)]['weight'])
    assert np.round(R_jc, 12) == np.round(jc, 12)


def test_make_fc_and_tvc():
    # Load parc data, make FC JC method
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.load_data('parcellation')
    r = tnet.make_functional_connectivity(returngroup=True)[0, 1]
    fc_files = tnet.get_selected_files(pipeline='functionalconnectivity')
    assert '_conn.tsv' in fc_files[0]
    assert len(fc_files) == 1
    R = tnet.parcellation_data_[0].transpose().corr().values[0, 1]
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time',
                 'postpro': 'standardize'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JC = tnet.tvc_data_[0].iloc[0].values[-1]
    # Load parc data, make FC JC method with FC dual weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time', 'weight-mean': 'from-subject-fc',
                 'weight-var': 'from-subject-fc'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JCw = tnet.tvc_data_[0].iloc[0].values[-1]
    # Load parc data, make FC JC method with FC mean weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time',
                 'weight-mean': 'from-subject-fc'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JCm = tnet.tvc_data_[0].iloc[0].values[-1]
    # Load parc data, make FC JC method with FC variance weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.derive_temporalnetwork({'method': 'jackknife', 'dimord': 'node,time',
                 'weight-var': 'from-subject-fc'}, update_pipeline=True, confound_corr_report=False)
    tnet.load_data('tvc')
    JCv = tnet.tvc_data_[0].iloc[0].values[-1]
    assert np.round(JCw, 15) == np.round((JC*r)+R, 15)
    assert np.round(JCv, 15) == np.round((JC*r), 15)
    assert np.round(JCm, 15) == np.round((JC)+R, 15)


# def test_communitydetection():
#     tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
#                              pipeline_subdir='tvc', bids_suffix='tvc', subjects='001', tasks='b', runs='alpha', raw_data_exists=False)
#     community_detection_params = {'resolution_parameter': 1,
#                                   'interslice_weight': 0, 'quality_function': 'ReichardtBornholdt2006'}
#     tnet.communitydetection(community_detection_params, 'temporal')
#     # Compensating for data not being in a versioen directory
#     tnet.set_pipeline('teneto_' + teneto.__version__)
#     tnet.load_community_data('temporal')
#     C = np.squeeze(tnet.community_data_)
#     assert C[0, 0] == C[1, 0] == C[2, 0]
#     assert C[3, 0] == C[4, 0] == C[5, 0]
#     assert C[0, 2] == C[1, 2] == C[2, 2] == C[3, 2]
#     assert C[4, 2] == C[5, 2]
#     assert C[3, 0] != C[0, 0]
#     assert C[4, 2] != C[0, 2]


def test_networkmeasure():
    # calculate and load a network measure
    bids_path = teneto.__path__[0] + '/data/testdata/dummybids/'
    pipeline = 'teneto_' + teneto.__version__
    tags = {'sub': '001', 'task': 'a', 'run': 'alpha'}
    tnet = teneto.TenetoBIDS(bids_path, pipeline=pipeline, pipeline_subdir='tvc', 
                            bids_suffix='tvcconn', bids_tags=tags, 
                            raw_data_exists=False)
    tnet.networkmeasures('volatility', {'calc': 'time'}, tag='time')
    tnet.load_data('temporalnetwork', measure='volatility', tag='time')
    assert tnet.temporalnetwork_data_['volatility'][0].shape == (19, 1)


# def test_timelockednetworkmeasure():
#     # calculate and load a network measure
#     tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto_' + teneto.__version__,
#                              pipeline_subdir='tvc', bids_suffix='tvc', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
#     tnet.networkmeasures('volatility', {'calc': 'time'}, save_tag='time')
#     tnet.load_data('temporalnetwork', measure='volatility')
#     vol = tnet.temporalnetwork_data_['volatility'][0].values
#     tnet.make_timelocked_events('volatility', 'testevents', [
#                                 1], [-1, 1], tag='time')
#     tnet.load_timelocked_data('volatility')
#     assert np.all(np.squeeze(tnet.timelocked_data_) ==
#                   np.squeeze(tnet.networkmeasure_[0, 0:3]))


def test_tnet_derive_with_removeconfounds():
    # load parc file with data
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    assert 'confound1' in alt
    assert 'confound2' in alt
    # Set the confounds
    tnet.set_confounds('confound1')
    # Remove confounds
    tnet.removeconfounds(transpose=True)
    f = tnet.get_selected_files()[0]
    f = f.replace('.tsv','.json')
    with open(f) as fs:  
        sidecar = json.load(fs)
    assert 'confoundremoval' in sidecar
    # Removing below tests due to errors caused by concurrent images.
    #tnet.derive_temporalnetwork({'method': 'jackknife'})
    # Make sure report directory exists
    #assert os.path.exists(teneto.__path__[0] + '/data/testdata/dummybids/derivatives/teneto_' + teneto.__version__ + '/sub-001/func/tvc/report')


def test_tnet_scrubbing():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    tnet.set_exclusion_timepoint('confound1', '>1', replace_with='nan')
    tnet.load_data('parcellation')
    dat = np.where(np.isnan(np.squeeze(tnet.parcellation_data_[0].values)))
    targ = np.array([[0, 0, 1, 1], [4, 5, 4, 5]])
    assert np.all(targ == dat)


def test_tnet_scrubbing_and_spline():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.load_data('parcellation')
    dat_orig = np.squeeze(tnet.parcellation_data_[0].values)
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    tnet.set_exclusion_timepoint('confound1', '>1', replace_with='cubicspline')
    tnet.load_data('parcellation')
    dat_scrub = tnet.parcellation_data_[0].values 
    targ = np.array([[0, 0, 1, 1], [4, 5, 4, 5]])
    # Make sure there is a difference
    assert np.sum(dat_scrub != dat_orig)
    # Show that the difference between the original data at scrubbed time point is larger in data_orig
    assert np.sum(
        np.abs(np.diff(dat_orig[0]))-np.abs(np.diff(dat_scrub[0]))) > 0
    # Future tests: test that the cubic spline is correct



def test_tnet_set_bad_files():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.load_data('parcellation')
    dat_orig = tnet.parcellation_data_[0].values
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    tnet.set_exclusion_file('confound2', '>0.5')
    assert len(tnet.bad_files) == 1
    assert tnet.bad_files[0] == tnet.BIDS_dir + 'derivatives/' + tnet.pipeline + \
        '/sub-001/func/' + tnet.pipeline_subdir + \
        '/sub-001_task-a_run-alpha_roi.tsv'

def test_tnet_make_parcellation():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='fmriprep',
                            bids_suffix='preproc', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    # Set the confound pipeline in fmriprep
    tnet.make_parcellation('gordon2014_333+sub-maxprob-thr25-1mm')
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='fmriprep',
                            bids_suffix='preproc', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.make_parcellation('gordon2014_333')
    tnet.load_data('parcellation')
    # Hard coded facts about dummy data
    assert tnet.parcellation_data_[0].shape == (2,333)

def test_tnet_checksidecar():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='fmriprep',
                            bids_suffix='preproc', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.make_parcellation('gordon2014_333')
    tnet.load_data('parcellation')
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '<=0', replace_with='nan')
    with open(teneto.__path__[0] + '/data/testdata/dummybids/derivatives/teneto_' + teneto.__version__ + '/sub-001/func/parcellation/sub-001_task-a_run-alpha_roi.json') as fs:  
        sidecar = json.load(fs)
    # Check both steps are in sidecar
    assert 'parcellation' in sidecar.keys()
    assert 'scrubbed_timepoints' in sidecar.keys()

def test_tnet_io():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='fmriprep',
                            bids_suffix='preproc', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.save_aspickle(teneto.__path__[0] + '/data/testdata/dummybids/teneosave.pkl')
    tnet2 = teneto.TenetoBIDS.load_frompickle(teneto.__path__[0] + '/data/testdata/dummybids/teneosave.pkl')
    assert tnet2.get_selected_files()==tnet.get_selected_files()

    
def test_tnet_scrubbing_and_exclusion_options():
    # <= 
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '<=0', replace_with='nan')
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_file('confound2', '<=1')
    # <
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound1', '<0', replace_with='nan')
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_file('confound2', '<1')
    # >=
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_timepoint('confound2', '>=2', replace_with='nan')
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/', pipeline='teneto-tests',
                             pipeline_subdir='parcellation', bids_suffix='roi', bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)
    tnet.set_confound_pipeline('fmriprep')
    tnet.set_exclusion_file('confound2', '>=1')

