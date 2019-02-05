Example TenetoBIDS Script
-------------------------

Below is an exmaple script for an entire analysis

>>> import teneto 
>>> # Number of concurrent processes to run
>>> njobs = 1
>>> #path to bids directory 
>>> data_path = './'
>>> # Define teneto object. The first argument is the path to the BIDS directory. The "pipeline" argument is the directory in the derivartives folder where the preprocessed data is. 
>>> tnet = teneto.TenetoBIDS(data_path,pipeline='fmriprep', bids_suffix='preproc', raw_data_exists=False)
>>> # --- PART 1 --- 
>>> # Preprocessing 
>>> tnet.confounds = None
>>> # Set confounds to remove
>>> confounds = ['X','Y','Z','RotX','RotY','RotZ', 'aCompCor00', 'aCompCor01', 'aCompCor02', 'aCompCor03', 'aCompCor04', 'aCompCor05']
>>> tnet.set_confounds(confounds)
>>> # This contains dictionary of information for additional preprocessing steps done by nilearn when making the parcellation. 
>>> nilearn_params = {'standardize': True, 'low_pass': 0.1,'high_pass': 0.01,'t_r':2}
>>> tnet.make_parcellation('gordon2014_333',removeconfounds=True,parc_params=nilearn_params)
>>> # Scrubbing
>>> # Remove all files that have a mean FWD above 0.2. This confound name should be equal to the confoun in get_confound_alternatives() (from fmriprep is FramewiseDisplacement (I think)) 
>>> tnet.set_exclusion_file('FramewiseDisplacement','>0.5')
>>> # Remove all timepoints with FWD > 0.2 and simulate with cubic spline. 
>>> # You can add a tol parameter. Which is a toloerance allowing x% to be above the threshold, other the subject is excluded. e.g. if tol=0.15, then if more than 15% of data is is greater than 0.2 - subject excluded.
>>> tnet.set_exclusion_timepoint('FramewiseDisplacement','>0.5','cubicspline')
>>> # See excluded subjects by (tnet.bad_files) 
>>> # Make static funcitonal connectivity (may not be needed)
>>> tnet.make_functional_connectivity()
>>> # Save checkpoint 
>>> tnet.save_aspickle(tnet.BIDS_dir + '/tnet_preprocess.pkl')
>>> # --- PART 2 --- 
>>> # DERIVE TVC
>>> # Run whichever method you want to use. Make sure only the last you do has update_pipeline=True.   
>>> tnet.derive_temporalnetwork({'method': 'jackknife', 'postpro': 'standardize'},confound_corr_report=False)
>>> # Save checkpoint
>>> tnet.save_aspickle(tnet.BIDS_dir + '/tnet_after_tvcderive')
>>> #This how you load (if needed). reload_object is set if I ever have to update the softare. 
>>> #tnet = teneto.TenetoBIDS.load_frompickle('./tnet_after_tvcderive.pkl',reload_object=True)
>>> #community detection per slice
>>> # --- PART 3 --- 
>>> # COMMUNITY DETECTION
>>> community_detection_params = {'resolution': 1, 'intersliceweight': 1} 
>>> tnet.communitydetection(community_detection_params,'temporal')
>>> # --- PART 4 --- 
>>> # NETWORK MEASURES 
>>> #Calculate the two network measures
>>> network_measures = ['temporal_degree_centrality']
>>> network_measure_params = [{}]
>>> tnet.networkmeasures(network_measures,network_measure_params)
