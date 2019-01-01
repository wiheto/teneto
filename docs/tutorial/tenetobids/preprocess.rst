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
