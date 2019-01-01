Teneto and BIDS: TenetoBIDS
---------------------------

The BIDS format for neuroimaging is a way to organize data. If your data is organized this way, then teneto can use this to efficiently process the data.  

The aim of TenetoBIDS is a class of functions which run on preprocessed fMRI data and run some additional preprocessing steps, time-varying connectivity derivation and quantify temporal network properties.
The compatiblity with the BIDS format allows for Teneto to run the analysis on all subjects/files/tasks with only a few lines of code.  

Prerequisites 
===================

The data used here should be from fMRI and preprocessed (e.g. fmriprep) and in the BIDS format. Whenever BIDS_dir is stated in the tutorial, it means the directory that points to the main BIDS directory.  

This preprocessed data should be in the .../BIDS_dir/derivatives/<pipeline> directory. For example, .../BIDS_dir/derivatives/fmriprep will contain the preprocessed files from fmriprep. The output from teneto will always be found in .../BIDS_dir/derivatives/teneto_<versionnumber>/
The output from Teneto is then ready to be placed in statistical models, machine learning algorithems and/or plotted. 

If is also useful to know where confounds files are in your preprocessed data. In fmriprep for example, they are called _confounds.csv. 

Data in tutorial
===================

For this tutorial, we will use some dummy data. This is included with the teneto package in teneto/teneto/testdata/dummybids/.

    >>> import teneto
    >>> import os
    >>> dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'

Here we see that there are two subjects.

    >>> os.listdir(dataset_path)
    ['participant.tsv', 'sub-001', 'derivatives', 'sub-002']

And in the derivatives we see two different derivatives that exist:

    >>> os.listdir(dataset_path + '/derivatives')
    ['teneto-tests', 'fmriprep']

What is relevant here is that there is fmriprep, which is where we have the dummy preprocessed data.

We can look at one of the subject:

    >>> os.listdir(dataset_path + '/derivatives/fmriprep/sub-001/func/')
    ['sub-001_task-a_run-beta_bold_preproc.nii.gz',
    'sub-001_task-a_run-beta_confounds.tsv',
    'sub-001_task-b_run-alpha_confounds.tsv',
    'sub-001_task-a_run-alpha_bold_preproc.nii.gz',
    'sub-001_task-b_run-alpha_bold_preproc.nii.gz',
    'sub-001_task-a_run-alpha_confounds.tsv']

Here we see there are two different tasks (a, b), and two different runs (alpha, beta).

There are both preproccessd nifti files (fMRI images) and tsv files which contain conounds (ending with the suffix _confounds).

Within the confound files we have two confounds: "confound1" and "confound2"

Note, here there is only one file with real data in. 

Defining TenetoBIDS
===================

The first step is to create a TenetoBIDS object. Here we always called such an object tnet. tnet is defined by calling the TenetoBIDS class. There are (at leas) two arguments 
that are needed, the BIDS_dir (i.e. the base directory) and the pipeline (the folder in the derivatives folder where the data is). 

    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='fmriprep', raw_data_exists=False)

Usually the argument raw_data_exists does not need to be passed, the reason for passing it is to ignore the data outside of the derivatives directories which is useful for how we have made this dummy data. But usually this can be ignored. 

Certain arguments can be passed when calling the TenetoBIDS class. 
For example, the data consists of tasks, runs, and subjects. To choose specific ones of these we can pass a dictionary of BIDS tags: 

    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='fmriprep',  bids_tags={'sub': '001', 'task': 'a', 'run': 'alpha'}, raw_data_exists=False)

This will mean that only subject 001, task a and run alpha are loaded. 

    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='fmriprep', raw_data_exists=False)
    >>> tnet.set_bids_tags({'sub': '001'})
    >>> tnet.set_bids_tags({'task': 'a', 'run': 'alpha'})

To see which files tnet identified within the BIDS directory and provide a summary of which files are currently selected, type: 

.. code-block:: python

    tnet.print_dataset_summary()

Which will print something like this: 

    |--- DATASET INFORMATION ---
    |--- Subjects ---
    |Number of subjects (selected): 1
    |Subjects (selected): 001
    |Bad subjects: 0
    |--- Tasks ---
    |Number of tasks (selected): 1
    |Tasks (selected): a
    |--- Runs ---
    |Number of runs (selected): 1
    |Rubs (selected): alpha
    |--- Sessions ---
    |Number of sessions (selected): 0
    |Sessions (selected): 
    |--- PREPROCESSED DATA (Pipelines/Derivatives) ---
    |Pipeline: fmriprep
    |--- SELECTED DATA ---
    |Numnber of selected files: 1
    |[list of files]

This helps provide some information about the data. 

If you ever want to retrive data from the BIDS structure, there are two that can be used. 
tnet.load_data() and tnet.get_selected_files(). The former will be demonstrated later. 

    >>> paths = tnet.get_selected_files(quiet=1)

Will produce a list of paths for the files that have been selected. The quiet argument states if this information is printed or not. 

Preprocessing steps
===================

Preprocessing software (e.g. fmriprep) often do the essential preprocessing steps for general fMRI analysis. 
However there are additional preprocessing steps that can be done (e.g. confound removal).
Teneto offers ways for this to be done. 

Teneto can do the following preprocessing steps. 

1. Remove confounds (e.g. global signal, movement, framewise displacement). 
2. Exclude files that exceed a certain confound. 
3. Remove time-points that have too much of a specified confound. 
4. Make regions of interest from a parcellation.  
5. Manually set bad files or bad subjects 

How to do each of these will be explained below. 

Removing confounds
******************

The procedure to remove confounds is in three steps (i) Make sure the confound files are being located. (ii) Check and specify which confounds you want to remove. (iii) Remove the confounds. 

Teneto looks for a '_confounds' file in each subject in the /derivatives/<pipeline>/ directory. However, if the confounds file are in another derivative directory (e.g. fmriprep)
and the current pipeline is derivatives/teneto_vx.y.z/, then it is possible to set the confound_pipeline (TenetoBIDS.set_confound_pipeline()). 

To check that the correct confound files are found, run TenetoBID.get_confound_files(). 

Once the correct confound files are identified, the next step is to choose which confounds should be removed from the data. To check which confounds are available, given the confound
files, run TenetoBIDS.get_confound_alternatives(). Then to set the confounds (e.g. 'FramewiseDisplacement' and 'GlobalSignal'), TenetoBIDS.set_confounds(['FramewiseDisplacement', 'GlobalSignal']). 
Then to remove the confounds run tnet.removeconfounds(). 

The removing of confounds uses nilearn.signal.clean. 

NOTE: run tnet.remove(transpose=True) if the input/saved data is of the dimensions node,time (then the saved data will also be node,time).  

Excluding subjects/files due to a confound
******************************************

The idea here is to find files where the average confound is greater or less than a specified value. A common confound where this is the case is FramewiseDisplacement 

To do this, call set_exclusion_file: 

.. code-block:: python

    tnet.set_exclusion_file(confound, exclusion_criteria, confound_stat)

The confound is the name of a column in the confoun value. The exclusion criteria is a string that represents the threshold for when a confound is considered "bad". And the confound_stat is whether the mean, median or std is above/below the exclusion_criteria. An example: 

.. code-block:: python

    tnet.set_exclusion_file('FramewiseDisplacement', '>0.2', 'mean')

This will mean that, if a file is greater than 0.2, it will be exclded from the data. 

The files that have been excluded can be found in tnet.bad_files().

Remove and simulate time-points due to a confound 
*************************************************

Instead of removing files from the analysis (or along with this strategy) another step is to remove the "bad" time-points due to a confound. Once they have been removed, they are either NaN values or they can be sinulated with a cubic spline. Example:

.. code-block:: python

    tnet.set_exclusion_timepoint('FramewiseDisplacement', '>0.2', replace_with='cubicspline')

Additionally, files can be marked as bad if a certain percentage of the time points are over that value. This is done with the argument tol. The below example marks files as bad if more than 20% of the data is above the specified exclusion criteria.  

.. code-block:: python

    tnet.set_exclusion_timepoint('FramewiseDisplacement', '>0.2', replace_with='cubicspline', tol=0.2)

Again, these subjects can be seen in tnet.bad_files() 

Make a parcellation
*******************

Teneto uses nilearn to make the parcellation. *more needed here*

Manually set bad files/bad subjects 
************************************

Files or subjects can be manually set as bad using tnet.set_bad_files() or tnet.set_bad_subjects(). 
For the bad subjects, the subject id should be specified. 
For the bad_file, the start of the BIDS file name should be specified (including tasks, runs, sessions etc. if appropriate).  

BIDS files generally have a json file that contains the metainformation. When setting a bad_subject or bad_file this will also be set in the 
json metadata and persists even if the BIDS object is redefined.

If you accidently write the wrong subject name in set_bad_subjects, this may 

Deriving time-varying representations 
======================================

To derive the time varying connectivity estimates, there are several available options in Teneto: 

- Sliding window 
- Tapered sliding window 
- Jackknife correlatoin 
- Spatial distance correlation 
- Multiply temporal derivatives 

Researchers have different preferences for different methods.

In TenetoBIDS, you can call the derive function and it will calculate the TVC estimates for you and placed
them in a tvc directory. 

In this example, we start out by selecting some dummy ROI data which is prespecfied in the teneto-tests directory. 

    >>> pipeline='teneto-tests'
    >>> data_directory = 'parcellation'
    >>> bids_tags = {'sub': '001', 'task': 'a', 'run': 'alpha'}
    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline=pipeline, pipeline_subdir=data_directory, bids_suffix='roi', bids_tags=bids_tags, raw_data_exists=False)

This contains 2 time-series which are 20 timepoints long. To see what we are working with, we can load the parcellation data and plot it. 

    >>> import matplotlib.pyplot as plt
    >>> tnet.load_data('parcellation')
    >>> tnet.parcellation_data_[0]
    >>> fig,ax = plt.subplots(1)
    >>> ax.plot(np.arange(1,21),tnet.parcellation_data_[0].transpose())
    >>> ax.set_ylabel('Amplitude')
    >>> ax.set_xlabel('Time')
    >>> plt.tight_layout()
    >>> fig.show() 

.. plot::

    import teneto
    import matplotlib.pyplot as plt 
    dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'
    pipeline='teneto-tests'
    data_directory = 'parcellation'
    bids_tags = {'sub': '001', 'task': 'a', 'run': 'alpha'}
    tnet = teneto.TenetoBIDS(dataset_path, pipeline=pipeline, pipeline_subdir=data_directory, bids_suffix='roi', bids_tags=bids_tags, raw_data_exists=False)
    tnet.load_data('parcellation')
    fig,ax = plt.subplots(1)
    ax.plot(np.arange(1,21),tnet.parcellation_data_[0].transpose())
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time')
    ax.set_xticks(np.arange(5,21,5))
    plt.tight_layout()
    fig.show() 

Let us say we want to apply the jackknife correlation method to this. To do this we just need to specify a dictionary of parameters which goes into teneto.derive.derive.
In the example below, we simply are saying we would to use the jackknife method and afterwards these estimates should be standerdized. 

    >>> derive_params = {'method': 'jackknife', 'postpro': 'standardize'}
    >>> tnet.derive(derive_params, confound_corr_report=False)
    ...

Setting confound_corr_report to true places a HTML showing histograms of each time-series each of the confounds so you can see how much the TVC is effected by them.

Now we have the time-varying estimates for each time-point, we can load and them by: 

    >>> tnet.load_data('tvc')

This produces a list of dataframes in tnet.tvc_data_. 

    >>> tnet.tvc_data_[0].head()
         i    j    t    weight
    0  0.0  1.0  0.0 -0.829939
    1  0.0  1.0  1.0  1.830899
    2  0.0  1.0  2.0 -0.278181
    3  0.0  1.0  3.0  0.108855
    4  0.0  1.0  4.0  0.417800

Where we see the columns for nodes (i,j), time-points (t) and the connectivity estimate (weight). 

These lists of connectivity estimates are for space purposes. They can be conveted to an array format (node,node,time) by 
calling teneto.TemporalNetwork (this may be included within TenetoBIDS at a later release): 

    >>> tvc = teneto.TemporalNetwork(from_df=tnet.tvc_data_[0])
    >>> conn_time_series = tvc.to_graphlet() 
    >>> conn_time_series
    (2, 2, 20)

Now as an array, we can easily visualise the connectivity time series between the two nodes. 

    >>> fig,ax = plt.subplots(1)
    >>> ax.plot(np.arange(1,21),conn_time_series[0,1,:])
    >>> ax.set_ylabel('Connectivity estimate (Jackknife)')
    >>> ax.set_xlabel('Time')
    >>> plt.tight_layout()
    >>> fig.show()     

.. plot::

    import teneto
    import matplotlib.pyplot as plt 
    import numpy as np
    dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'
    pipeline='teneto-tests'
    data_directory = 'parcellation'
    bids_tags = {'sub': '001', 'task': 'a', 'run': 'alpha'}
    tnet = teneto.TenetoBIDS(dataset_path, pipeline=pipeline, pipeline_subdir=data_directory, bids_suffix='roi', bids_tags=bids_tags, raw_data_exists=False)
    derive_params = {'method': 'jackknife', 'postpro': 'standardize'}
    tnet.derive(derive_params, confound_corr_report=False)  
    tnet.load_data('tvc')  
    tvc = teneto.TemporalNetwork(from_df=tnet.tvc_data_[0])
    conn_time_series = tvc.to_graphlet() 
    fig,ax = plt.subplots(1)
    ax.plot(np.arange(1,21),conn_time_series[0,1,:])
    ax.set_ylabel('Connectivity estimate (Jackknife)')
    ax.set_xlabel('Time')
    ax.set_xticks(np.arange(5,21,5))
    plt.tight_layout()
    fig.show()     


Community Detection
===================

*Currently disabled, being rewritten*

Temporal network properties
===========================

tnet.networkmeasures()

Timelocked data 
=================

*Currently disabled, being rewritten*

Load saved data
=================

You can save you progress by using save_pickle 

.. code-block:: python

    tnet.save_aspickle(dataset_path + '/tenetoobj.pkl')

Then to load it you just need to write

.. code-block:: python

    tnet = teneto.TenetoBIDS.load_frompickle(dataset_path + '/tenetoobj.pkl')


BIDS suffixes used in tenetoBIDS 
========================

=====     =====  
Suffix    Description 
=====     ===== 
_conn     connectivity matrix
_tvcconn  time-varying connectivity matrix
_roi      regions of interest
_tnet     temporal network estimate
=====     =====

File formats are all .tsv files. 

Note: These are not approved in the BIDS derivaives/connectivity at the moment as the connectivity is not out.


Information in JSON sidecars
========================

The JSON sidecars offer information about the different files. 
All the input parameters for functions are listed there. 


