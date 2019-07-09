Preprocessing steps
===================

Preprocessing software (e.g. fmriprep) do the essential
initial preprocessing steps for general fMRI analysis.
However there are additional preprocessing steps
that need to be done (e.g. confound removal).
Teneto offers ways for this to be done.

Teneto can do the following preprocessing steps.

1. Remove confounds (e.g. global signal, movement, framewise displacement).
2. Exclude files that exceed a certain confound.
3. Remove time-points that have too much of a specified confound.
4. Make regions of interest from a parcellation.
5. Manually set bad files or bad subjects.

How to do each of these will be explained below.

fmriprep version greater than 1.14 is for current versions of TenetoBIDS.

Removing confounds
******************

The procedure to remove confounds is in three steps:

1. Make sure the confound files are being located.
2. Check and specify which confounds you want to remove.
3. Remove the confounds.

Teneto looks for a confounds file in each subject in the
/derivatives/<pipeline>/ directory.
The confounds file ends with the suffix __regressors.tsv_
and includes _desc-confounds_.

However, if the confounds file is in another derivatives directory
(e.g. fmriprep) than your active pipeline (e.g. teneto_x.y.z),
then it is possible to specify a "condounds pipeline". Just run:

    >>> TenetoBIDS.set_confound_pipeline('fmriprep')
    ...

To check that the correct confound files are found,
run TenetoBID.get_confound_files().

Once the correct confound files are identified,
the next step is to choose which confounds should be removed from the data.
To check which confounds are available, given the confound
files, run:

    >>> TenetoBIDS.get_confound_alternatives().
    ...

This will display a list of confound names found in the confounds file.
Then to set the confounds to be removed
(e.g. 'framewise_displacement' and 'global_signal'),

    >>> tnet.set_confounds(['framewise_displacement', 'global_signal'])
    ...

Then to remove the confounds:

    >>> tnet.removeconfounds()
    ...

The removing of confounds uses nilearn.signal.clean
and any additional arguments can be called using this function
by passing a dictionary argument called clean_params
(e.g. apply high or low pass filter, detrend or standardize).

    >>> clean_params = {'high_pass': 0.01,
                        't_r': 2}
    >>> tnet.removeconfounds(clena_params=clean_parmas)

NOTE: run tnet.remove(transpose=True) if the input/saved data
is of the dimensions node,time (and the saved data as node,time).

Excluding subjects/files due to a confound
******************************************

Some subjects should have an entire run excluded because
one of the possible confounds is repeatedly low/high for that run.
For example, if a subject moves too much,
we will want to exclude them from further analysis.

In order to exclude an entire file (e.g. a run), we need:

1. A confound (e.g. framewise_displacement)
2. An exclusion critera (e.g. '>0.5')
3. A statistic of the confound to test the an exclusion criteria against (e.g. 'mean').

The above examples will say: "exclude all files that
have their mean framewise_displacement greater than 0.5".

To do this, call set_exclusion_file:

    >>> tnet.set_exclusion_file('framewise_displacement', '>0.2', 'mean')

The confound_stat can be mean, median or std.

The files that have been excluded can be found in tnet.bad_files()
and are also recorded in the json sidecar.

Remove and simulate time-points due to a confound
*************************************************

Instead of removing files from the analysis (or along with this strategy)
another step is to remove the "bad" time-points due to a confound. 
Once they have been removed, they are either NaN values or they can be sinulated with a cubic spline. 

Example:

.. code-block:: python

    tnet.set_exclusion_timepoint('FramewiseDisplacement', '>0.2', replace_with='cubicspline')

Additionally, files can be marked as bad if a certain percentage of the time points are over that value. This is done with the argument tol. The below example marks files as bad if more than 20% of the data is above the specified exclusion criteria.  

.. code-block:: python

    tnet.set_exclusion_timepoint('FramewiseDisplacement', '>0.2', replace_with='cubicspline', tol=0.2)

Again, these subjects can be seen in tnet.bad_files()

Make a parcellation
*******************

Teneto uses TemplateFlow's repository of atlases.
Any atlas available in `TemplateFlow <https://github.com/templateflow/templateflow/>`_
can be used to create a parcellation in Teneto.

If you are using fmriprep, then the default voxel output is in MNI152NLin2009cAsym space.
(you can check this by looking at the space- tag of your output files).

>>> tnet.make_parcellation(atlas='Schaefer2018', atlas_desc='400Parcels7Networks')

This will download the Schaefer2018 atlas of 400 parcels using TemplateFlow.
Then the time series of all parcels will be extracted and placed in:
/derivatives/teneto_*/parcellation/

If you use a different space then MNI152NLin2009cAsym,
this can be specified with the template arugment.

Teneto makes use of nilearn's function:
`NiftiLabelsMasker https://nilearn.github.io/modules/generated/nilearn.input_data.NiftiLabelsMasker.html`_ 
and any argument to this function can be passed in a dictionary called parc_params.

>>> atlas = 'Schaefer2018'
>>> atlastype = '400Parcels7Networks'
>>> parc_params = {'standardize': True, 
                'detrend': True, 
                'low_pass': 0.1,
                'high_pass': 0.008,
                't_r': 0.8} 
>>> tnet.make_parcellation(atlas=atlastype, atlas_desc=atlastype)


Manually set bad files/bad subjects
************************************

Files or subjects can be manually set as bad using tnet.set_bad_files() or tnet.set_bad_subjects().
For the bad subjects, the subject id should be specified.
For the bad_file, the start of the BIDS file name should be specified
(including tasks, runs, sessions etc. if appropriate).

BIDS files can have a json "sidecar" that contains the meta-information about the file.
When setting a bad_subject or bad_file this will also be set in the
json metadata and persists even if the BIDS object is redefined.

When setting a subject as "bad" you can pass the argument _reason_
which will record what the reason was in the json sidecar. 

If you accidently write the wrong subject name in set_bad_subjects,
can be erased by using the opps=True and reason='last' and it will remove the last reason subjects were considered bad. 
