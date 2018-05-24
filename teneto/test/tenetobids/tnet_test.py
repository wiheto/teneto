import teneto 
import numpy as np 

def test_tnet_derive(): 
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.load_parcellation_data()
    tnet.derive({'method':'jackknife','dimord':'node,time',},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    R_jc = teneto.misc.corrcoef_matrix(np.squeeze(tnet.parcellation_data_)[:,1:])[0][0,1]
    assert np.round(R_jc,12) == np.round(tnet.tvc_data_[0,0,1,0]*-1,12)

def test_make_fc():
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.load_parcellation_data()
    r = tnet.make_functional_connectivity(returngroup=True)[0,1] 
    R = teneto.misc.corrcoef_matrix(np.squeeze(tnet.parcellation_data_))[0][0,1]
    tnet.derive({'method':'jackknife','dimord':'node,time','postpro':'standardize'},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    JC = tnet.tvc_data_[0,0,1,:]
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-mean':'from-subject-fc','weight-var':'from-subject-fc'},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    JCw = tnet.tvc_data_[0,0,1,:]
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.make_functional_connectivity() 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-mean':'from-subject-fc'},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    JCm = tnet.tvc_data_[0,0,1,:]
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.make_functional_connectivity() 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-var':'from-subject-fc'},update_pipeline=True,confound_corr_report=True)
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

