Teneto and BIDS: TenetoBIDS
---------------------------

(This tutorial is under construction)

The BIDS format for neuroimaging is a way to organize data. If your data is organized this way, then teneto can use this to efficiently process the data.  

The aim of TenetoBIDS is a class of functions which run on preprocessed fMRI data and run some additional preprocessing steps, time-varying connectivity derivation and quantify temporal network properties.
The compatiblity with the BIDS format allows for Teneto to run the analysis on all subjects/files/tasks with only a few lines of code.  

Prerequisites 
================= 

The data used here should be from fMRI and preprocessed (e.g. fmriprep) and in the BIDS format. Whenever BIDS_dir is stated in the tutorial, it means the directory that points to the main BIDS directory.  

This preprocessed data should be in the .../BIDS_dir/derivatives/<pipeline> directory. For example, .../BIDS_dir/derivatives/fmriprep will contain the preprocessed files from fmriprep. The output from teneto will always be found in .../BIDS_dir/derivatives/teneto_<versionnumber>/
The output from Teneto is then ready to be placed in statistical models, machine learning algorithems and/or plotted. 

If is also useful to know where confounds files are in your preprocessed data. In fmriprep for example, they are called _confounds.csv. 

Defining TenetoBIDS
===================

The first step is to define a TenetoBIDS object. Here we always called such an object tnet. tnet is defined by calling the TenetoBIDS class. There are (at leas) two arguments 
that are needed, the BIDS_dir (i.e. the base directory) and the pipeline (the folder in the derivatives folder where the data is). 

.. code-block:: python

    tnet = teneto.TenetoBIDS('/directory_to_bids/', pipeline='fmriprep')

Certain arguments can be passed when calling the TenetoBIDS class. For example, if you only want to use a few subjects and a specific task you run: 

.. code-block:: python

    tnet = teneto.TenetoBIDS('/directory_to_bids/', pipeline='fmriprep', subjects=['01','02','03'], tasks='mytask1')

This will mean that only sub-01, sub-2, sub-03 are used and task-mytask1. Alternativelt the same can be achieved by splitting this up into subfuncitons  

.. code-block:: python

    tnet = teneto.TenetoBIDS('/directory_to_bids/')
    tnet.set_pipeline('fmriprep')
    tnet.set_subjects(['01','02','03'])
    tnet.set_tasks('mytask1')

To see which files tnet identified within the BIDS directory and provide a summary of which files are currently selected, type: 

.. code-block:: python

    tnet.print_dataset_summary()

Which will print general information about what the object can find in the dataset and which have been selected. 

Alternatively you can see which files are currently selected and (in most instances) these will be acted upon if calling a TenetoBIDS function.  

.. code-block:: python

    tnet.get_selected_files()

If there is a _confounds file within the preprocessed file (or an independent derivatives folder that is specified with TenetoBIDS.set_confound_pipeline())

.. code-block:: python 
    tnet.get_confound_files()

Note: while other preprocessing pipelines are compatible with teneto, all functions have been tested on fmriprep and some (e.g. confound removal) may not work. This will be made better when the BIDS derivative format is finalized. 


Preprocessing steps
===================

Teneto can do the following preprocessing steps. 

1. Remove confounds (e.g. global signal, movement, framewise displacement). 
2. Exclude subjects that have too much of a specified  confound. 
3. Remove time-points or simulation removed time-points that have too much of a specified confound. 
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

Files or subjects can be manually set as bad using tnet.set_bad_files() or tnet.set_bad_subjects(). For the bad subjects, the subject id should be specified. For the bad_file, the start of the BIDS file name should be specified (including tasks, runs, sessions etc. if appropriate).  

Deriving time-varying representations 
======================================

tnet.derive()

Community Detection
===================

tnet.communiydetection() 

Temporal network properties
===========================

tnet.networkmeasures()

Timelocked data 
=================

tnet.make_timelocked_data()

Load saved data
=================

tnet.save_aspickle()
