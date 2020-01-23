
#%% [markdown]
#
# # TenetoBIDS
#
# TenetoBIDS allows use of Teneto functions to analyse entire datasets in just a few lines of code.
# The output from Teneto is then ready to be placed in statistical models, machine learning algorithms and/or plotted.
#
# ## Prerequisites
#
# To use *TenetoBIDS* you need preprocessied fMRI data in the [BIDS format](https://github.com/bids-standard/bids-specification). 
# It is tested and optimized for [fMRIPrep](https://fmriprep.readthedocs.io/en/stable/) but other preprocessing software following BIDS should (in theory) work too.
# For fMRIPrep V1.4 or later is requiresd.
# This preprocessed data should be in the ~BIDS_dir/derivatives/<pipeline> directory.
# The output from teneto will always be found in .../BIDS_dir/derivatives/ in directories that begin with teneto- (depending on the function you use).
#
# ## Contents of this tutorial
# 
# This tutorial will run a complete analysis on some test data.
# 
# For this tutorial, we will use some dummy data which is included with teneto.
# This section details what is in this data.
# 
# %%
import teneto
import os
dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'
print(os.listdir(dataset_path))
print(os.listdir(dataset_path + '/derivatives'))

# %% [markdown]
# From the above we can see that there are two subjects in our dataset,
# and there is an fmriprep folder in the derivatives section.
# Only subject 1 has any dummy data, so we will have to select subject 1.

# %% [markdown]
#
## A complete analysis
#
# Below is an entire analysis on this test data. We will go through each step after it.

# %%

# Imports.
from teneto import TenetoBIDS
from teneto import __path__ as tenetopath
import numpy as np
# Set the path of the dataset.
datdir = tenetopath[0] + '/data/testdata/dummybids/'

# Step 1: 
bids_filter = {'subject': '001', 
               'run': 1,
               'task': 'a'}
tnet = TenetoBIDS(datdir, selected_pipeline='fmriprep', bids_filter=bids_filter, exist_ok=True)

# Step 2: create a parcellation
parcellation_params = {'atlas': 'Schaefer2018',
                       'atlas_desc': '100Parcels7Networks',
                       'parc_params': {'detrend': True}}
tnet.run('make_parcellation', parcellation_params)

# Step 3: Regress out confounds 
remove_params = {'confound_selection': ['confound1']}
tnet.run('remove_confounds', remove_params)

# Step 4: Additonal preprocessing 
exclude_params = {'confound_name': 'confound1',
                   'exclusion_criteria': '<-0.99'}
tnet.run('exclude_runs', exclude_params)
censor_params = {'confound_name': 'confound1',
                   'exclusion_criteria': '<-0.99',
                   'replace_with': 'cubicspline',
                   'tol': 0.25}
tnet.run('censor_timepoints', censor_params)

# Step 5: Calculats time-varying connectivity
derive_params = {'params': {'method': 'jackknife',
                            'postpro': 'standardize'}}
tnet.run('derive_temporalnetwork', derive_params)

# Step 6: Performs a binarization of the network 
binaraize_params = {'threshold_type': 'percent',
                    'threshold_level': 0.1}
tnet.run('binarize', binaraize_params)

# Step 7: Calculate a network measure
measure_params = {'distance_func': 'hamming'}
tnet.run('volatility', measure_params)

# Step 8: load data
vol = tnet.load_data()
print(vol)

#%% [markdown]
#
# ## Big Picture
# 
# While the above code may seem overwhelming at first.
# It is quite little code for what it does.
# It starts off with nifti images and ends with a single measure about
# a time-varying connectivity estimate of the network.
#
# There is one recurring theme used in the code above:
#
# `tnet.run(function_name, function_parameters)`
# 
# function_name is a string and function_parameters is a dictionary 
# function_name can be most functions in teneto if the data is in the correct format.
# function_parameters are the inputs to that funciton.
# You never need to pass the input data (e.g. time series or network),
# or any functions that have a `sidecar` input.
# 
# TenetoBIDS will also automaticlally try and find a confounds file in the derivatives when needed,
# so this does not need to be specified either.
#
# Once you have grabbed the above, the rest is pretty straight forward. But we will go through each step in turn.
#  
#%% [markdown]
#
# ## Step 1 - defining the TenetoBIDS object. 
#

#%%
# Set the path of the dataset.
datdir = tenetopath[0] + '/data/testdata/dummybids/'
# Step 1: 
bids_filter = {'subject': '001', 
               'run': 1,
               'task': 'a'}
tnet = TenetoBIDS(datdir, selected_pipeline='fmriprep', bids_filter=bids_filter, exist_ok=True)

#%% [markdown]
# ### selected_pipeline
#
# **This states where teneto will go looking for files. This says it should look in the fmriprep derivative directory.
# (i.e. in: datadir + '/derivatives/fmriprep/'). 
# 
# ### bids_filter
#
# teneto uses [pybids](https://github.com/bids-standard/pybids/) to select different files.
# The `bids_filter` argument is a dicitonary of arguments that get passed into the `BIDSLayout.get`.
# In the example above, we are saying we want subject 001, run 1 and task a.
# If no bids_filter is provided, all data within the derivatives folder will be aanlaysed.
# 
# ### exist_ok
# 
# This checks that it is ok to overwrite any previous calculations.
# The output data is saved in a new directory, but if the function has already been run before,
# this will need to be set to True if overwriting the old data is ok.
# If False (the default) then an error will be thrown if teneto outputs already exist. 

#%% [markdown]
# We can now look at what files are selected that will be run on the next step. 

#%%
tnet.get_selected_files()

#%% [markdown]
# If there are files here you do not want, you can add to the bids filter with `tnet.update_bids_filter`
# Or, you can set tnet.bids_filter to a new dictionary if you want.

#%% [markdown]
# Next you might want to see what functions you can run on these selected files.
# The following will specify what functions can be run specifically on the selected data.
# If you want all options, you can add the `for_selected=False`.
#%% 
tnet.get_run_options()

#%% [markdown]
# The output here (exclude_runs and make_parcellation) says which functions that, with the selected files, can be called in tnet.run.
# Once different functions have been called, the options change.

#%% [markdown]
# ## Step 2 Calling the run function to make a parcellation.
#
# When selecting preprocessed files, these will often be nifti images.
# From these, we want to make timeseries. This can be done with :py:func:`.make_parcellation`.
# This function uss [TemplateFlow](https://github.com/templateflow/templateflow/) atlases to make the parcellation.

#%%
parcellation_params = {'atlas': 'Schaefer2018',
                       'atlas_desc': '100Parcels7Networks',
                       'parc_params': {'detrend': True}}
tnet.run('make_parcellation', parcellation_params)

#%% [markdown]
# The `atlas` and `atlas_desc` are used to identify TemplateFlow atlases. 
#
# Teneto uses nilearn's [NiftiLabelsMasker](https://nilearn.github.io/modules/generated/nilearn.input_data.NiftiLabelsMasker.html)
# to mark the parcellation.
# Any arguments to this function (e.g. preprocessing steps) can be passed in the argument using 'parc_params' (here detrend is used).

#%% [markdown]
# ## Step 3 Regress out confounds

#%%
remove_params = {'confound_selection': ['confound1']}
tnet.run('remove_confounds', remove_params)

#%% [markdown]
# Confounds can be removed by calling :py:func:`.remove_confounds`.
#
# The confounds tsv file is automatically located as long as it is in a derivatives folder and that there is only one
# 
# Here 'confound1' is a column namn in the confounds tsv file.
#
# Similarly to make parcellation, it uses nilearn ([nilean.signal.clean](https://nilearn.github.io/modules/generated/nilearn.signal.clean.html).
# `clean_params` is a possible argument, like `parc_params` these are inputs to the nilearn function.
#


#%% [markdown]
# ## Step 4: Additonal preprocessing 

#%%
exclude_params = {'confound_name': 'confound1',
                   'exclusion_criteria': '<-0.99'}
tnet.run('exclude_runs', exclude_params)
censor_params = {'confound_name': 'confound1',
                   'exclusion_criteria': '<-0.99',
                   'replace_with': 'cubicspline',
                   'tol': 0.25}
tnet.run('censor_timepoints', censor_params)


#%% [markdown]

# These two calls to tnet.run exclude both time-points and runs which are problematic. 
# The first, exclude_runs, rejects any run where the mean of confound1 is less than 0.99. 
# Excluded runs will no longer be part of the loaded data in later calls of tnet.run().
# 
# Centoring timepoints here says that whenever there is a time-point that is less than 0.99 it will be "censored" (set to not a number).
# We have also set argument replace_with to cubicspline. This means that the values that have censored now get simulated using a cubic spline.
# The parameter tol says what percentage of time-points are allowed to be censored before the run gets ignored. 

#%% [markdown] 
# ## Step 5: Calculats time-varying connectivity
#
# The code below now derives time-varying connectivity matrices.
# There are multiple different methods that can be called.
# See teneto.timeseries.derive_temporalnetwork for more options.
# 

#%%
derive_params = {'params': {'method': 'jackknife',
                            'postpro': 'standardize'}}
tnet.run('derive_temporalnetwork', derive_params)

#%% [markdown]
# ## Step 6: Performs a binarization of the network 
# 
# Once you have a network representation,
# there are multiple ways this can be transformed.
# One example, is to binarize the network so all values are 0 or 1.
# The code below converts the top 10% of edges to 1s, the rest 0.
#%%
binaraize_params = {'threshold_type': 'percent',
                    'threshold_level': 0.1}
tnet.run('binarize', binaraize_params)

#%% [markdown]
# ## Step 7: Calculate a network measure
#
# We are now ready to calculate a property of the temproal network.
# Here we calculate volatility (i.e. how much the network changes per time-point).
# This generates one value per subject. 
#%% 
measure_params = {'distance_func': 'hamming'}
tnet.run('volatility', measure_params)

#%% [markdown]
# ## Step 8: load data

#%%
vol = tnet.load_data()
print(vol)

#%% [markdown]
# Now that we have a measure of volatility for the network.
# We can now load it, and view the measure.