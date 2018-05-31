import teneto 
import numpy as np 
import os 


def test_tnet_derive(): 
    # load parc file with data
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.load_parcellation_data()
    tnet.set_confound_pipeline('fmriprep')
    # Turn the confound_corr_report to True once matplotlib works withconcurrent
    tnet.derive({'method':'jackknife','dimord':'node,time'},update_pipeline=True,confound_corr_report=False)
    tnet.load_tvc_data()
    R_jc = teneto.misc.corrcoef_matrix(np.squeeze(tnet.parcellation_data_)[:,1:])[0][0,1]
    assert np.round(R_jc,12) == np.round(tnet.tvc_data_[0,0,1,0]*-1,12)

def test_make_fc_and_tvc():
    # Load parc data, make FC JC method
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.load_parcellation_data()
    r = tnet.make_functional_connectivity(returngroup=True)[0,1] 
    fc_files = tnet.get_functional_connectivity_files() 
    assert 'roi_fc' in fc_files[0]
    assert len(fc_files) == 1
    R = teneto.misc.corrcoef_matrix(np.squeeze(tnet.parcellation_data_))[0][0,1]
    tnet.derive({'method':'jackknife','dimord':'node,time','postpro':'standardize'},update_pipeline=True,confound_corr_report=False)
    tnet.load_tvc_data()
    JC = tnet.tvc_data_[0,0,1,:]
    # Load parc data, make FC JC method with FC dual weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-mean':'from-subject-fc','weight-var':'from-subject-fc'},update_pipeline=True,confound_corr_report=False)
    tnet.load_tvc_data()
    JCw = tnet.tvc_data_[0,0,1,:]
    # Load parc data, make FC JC method with FC mean weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-mean':'from-subject-fc'},update_pipeline=True,confound_corr_report=False)
    tnet.load_tvc_data()
    JCm = tnet.tvc_data_[0,0,1,:]
    # Load parc data, make FC JC method with FC variance weighting
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-var':'from-subject-fc'},update_pipeline=True,confound_corr_report=False)
    tnet.load_tvc_data()
    JCv = tnet.tvc_data_[0,0,1,:]
    assert all(JCw==(JC*r)+R)
    assert all(JCv==(JC*r))
    assert all(JCm==(JC)+R)


    
    

def test_communitydetection(): 
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='tvc',last_analysis_step='tvc',subjects='001',tasks='b',runs='alpha',raw_data_exists=False) 
    community_detection_params = {'resolution_parameter': 1, 'interslice_weight': 0, 'quality_function': 'ReichardtBornholdt2006'} 
    tnet.communitydetection(community_detection_params,'temporal')
    # Compensating for data not being in a versioen directory
    tnet.set_pipeline('teneto_' + teneto.__version__)
    tnet.load_community_data('temporal')
    C = np.squeeze(tnet.community_data_)
    assert C[0,0] == C[1,0] == C[2,0]
    assert C[3,0] == C[4,0] == C[5,0]
    assert C[0,2] == C[1,2] == C[2,2] == C[3,2]
    assert C[4,2] == C[5,2]
    assert C[3,0] != C[0,0] 
    assert C[4,2] != C[0,2] 



def test_networkmeasure(): 
    # calculate and load a network measure
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto_' + teneto.__version__,pipeline_subdir='tvc',last_analysis_step='tvc',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.networkmeasures('volatility')
    tnet.load_network_measure('volatility')
    assert len(tnet.networkmeasure_) == 1


def test_tnet_derive_with_removeconfounds(): 
    # load parc file with data
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    # Set the confound pipeline in fmriprep 
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    assert 'confound1' in alt
    assert 'confound2' in alt
    #Set the confounds 
    tnet.set_confounds('confound1')
    #Remove confounds
    tnet.removeconfounds(transpose=True)
    assert tnet.last_analysis_step == 'clean'
    tnet.derive({'method': 'jackknife'})
    # Make sure report directory exists
    assert os.path.exists(teneto.__path__[0] + '/data/testdata/dummybids/derivatives/teneto_' + teneto.__version__ + '/sub-001/func/tvc/report')

def test_tnet_scrubbing():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    # Set the confound pipeline in fmriprep 
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    tnet.set_exclusion_timepoint('confound1','>1',replace_with='nan')
    tnet.load_parcellation_data(tag='scrub')
    dat = np.where(np.isnan(np.squeeze(tnet.parcellation_data_)))
    targ = np.array([[0, 0, 1, 1],[4,5,4,5]])
    assert np.all(targ == dat)

def test_tnet_scrubbing_and_spline():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    # Set the confound pipeline in fmriprep 
    tnet.load_parcellation_data()
    dat_orig = np.squeeze(tnet.parcellation_data_)
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    tnet.set_exclusion_timepoint('confound1','>1',replace_with='cubicspline')
    tnet.load_parcellation_data(tag='scrub')
    dat_scrub = np.squeeze(tnet.parcellation_data_)
    targ = np.array([[0, 0, 1, 1],[4,5,4,5]])
    # Make sure there is a difference 
    assert np.sum(dat_scrub != dat_orig)
    # Show that the difference between the original data at scrubbed time point is larger in data_orig
    assert np.sum(np.abs(np.diff(dat_orig[0]))-np.abs(np.diff(dat_scrub[0]))) > 0
    # Future tests: test that the cubic spline is correct


def test_tnet_set_bad_files():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    # Set the confound pipeline in fmriprep 
    tnet.load_parcellation_data()
    dat_orig = np.squeeze(tnet.parcellation_data_)
    tnet.set_confound_pipeline('fmriprep')
    alt = tnet.get_confound_alternatives()
    tnet.set_exclusion_file('confound2','>0.5')
    assert len(tnet.bad_files) == 1 
    assert tnet.bad_files[0] == tnet.BIDS_dir + 'derivatives/' + tnet.pipeline + '/sub-001/func/' + tnet.pipeline_subdir + '/sub-001_task-a_run-alpha_bold_preproc_roi' 

