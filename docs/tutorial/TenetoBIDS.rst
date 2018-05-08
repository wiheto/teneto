Teneto and BIDS: TenetoBIDS
---------------------------

(This tutorial is under construction)

The BIDS format for neuroimaging is a way to organize neuroimaging data. 

The aim of TenetoBIDS is a class of functions which can take preprocessed fMRI data and run some additional preprocessing steps, time-varying connectivity derivation and quantify temporal network properties.
The compatiblity with the BIDS format allows for Teneto to run the analysis on all subjects/files/tasks with only a few lines of code.  

The data used here should be from fMRI and preprocessed (e.g. fmriprep). In the BIDS format this will be found in the derivatives folder. 
For example, .../BIDS_dir/derivatives/fmriprep will contain the preprocessed files from fmriprep. The output from teneto will always be found in .../BIDS_dir/derivatives/teneto_<versionnumber>/
The output from Teneto is then ready to be placed in statistical models, machine learning algorithems and/or plotted. 

Defining TenetoBIDS
===================

.. code-block:: python

    tnet = teneto.TenetoBIDS('/directory_to_bids/',pipeline='fmriprep')

Certain arguments can be passed when calling the TenetoBIDS class. For example, if you only want to use a few subjects and a specific task you run: 

.. code-block:: python

    tnet = teneto.TenetoBIDS('/directory_to_bids/',pipeline='fmriprep',subjects=['01','02','03'],tasks='mytask1')

This will mean that only sub-01, sub-2, sub-03 are used and task-mytask1. 

To see which files tnet identified within the BIDS directory and provide a summary of which files are currently selected, type: 

.. code-block:: python

    tnet.print_dataset_summary()

Which will print general information about what the object can find. 

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

.. code-block:: python


Remove and simulate time-points due to a confound 
*************************************************


Make a parcellation
*******************


Manually set bad files/bad subjects 
************************************


Deriving time-varying representations 
======================================


Community Detection
===================


Temporal network properties
===========================


Timelocked data 
=================


Load saved data
=================

