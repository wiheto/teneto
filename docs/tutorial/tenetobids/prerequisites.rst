Prerequisites 
===================

The data used here should be from fMRI and preprocessed (e.g. fmriprep) and in the BIDS format. Whenever BIDS_dir is stated in the tutorial, it means the directory that points to the main BIDS directory.  

This preprocessed data should be in the .../BIDS_dir/derivatives/<pipeline> directory. For example, .../BIDS_dir/derivatives/fmriprep will contain the preprocessed files from fmriprep. The output from teneto will always be found in .../BIDS_dir/derivatives/teneto_<versionnumber>/
The output from Teneto is then ready to be placed in statistical models, machine learning algorithems and/or plotted. 

If is also useful to know where confounds files are in your preprocessed data. In fmriprep for example, they are called _confounds.csv. 
