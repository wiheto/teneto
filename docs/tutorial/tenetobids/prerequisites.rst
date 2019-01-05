Prerequisites 
===================

To use *TenetoBIDS* you need two things: 

1. fMRI data in the BIDS format. 
2. Running some preprocessing on the fMRI data (e.g. fmriprep). 

Whenever BIDS_dir is stated in the tutorial, it means the directory that points to the main BIDS directory.  

This preprocessed data should be in the .../BIDS_dir/derivatives/<pipeline> directory (which is the automatic directory for preprocessing software). For example, .../BIDS_dir/derivatives/fmriprep will contain the preprocessed files from fmriprep. The output from teneto will always be found in .../BIDS_dir/derivatives/teneto_<versionnumber>/
The output from Teneto is then ready to be placed in statistical models, machine learning algorithems and/or plotted. 

If is also useful to know where confounds files are in your preprocessed data. These have the suffix _confounds.csv and should be the directory where the preprocessed data is. 
