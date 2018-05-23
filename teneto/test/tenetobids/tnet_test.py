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
    tnet.derive({'method':'jackknife','dimord':'node,time'},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    JC = tnet.tvc_data_[0,0,1,:]
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-mean':True,'weight-var':True},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    JCw = tnet.tvc_data_[0,0,1,:]
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.make_functional_connectivity() 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-mean':True},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    JCm = tnet.tvc_data_[0,0,1,:]
    tnet = teneto.TenetoBIDS(teneto.__path__[0] + '/data/testdata/dummybids/',pipeline='teneto-tests',pipeline_subdir='parcellation',last_analysis_step='roi',subjects='001',tasks='a',runs='alpha',raw_data_exists=False) 
    tnet.make_functional_connectivity() 
    tnet.derive({'method':'jackknife','dimord':'node,time','weight-var':True},update_pipeline=True,confound_corr_report=True)
    tnet.load_tvc_data()
    JCv = tnet.tvc_data_[0,0,1,:]

    assert tnet.tvc_data[0,0,1,:].std() = 1
    R_jc = teneto.misc.corrcoef_matrix(np.squeeze(tnet.parcellation_data_)[:,1:])[0][0,1]
    R_jc2 = teneto.misc.corrcoef_matrix(np.squeeze(tnet.parcellation_data_)[:,:])[0][0,1]
    
