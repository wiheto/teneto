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

