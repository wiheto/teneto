Data in tutorial
===================

For this tutorial, we will use some dummy data
which is included with teneto.
To view the files we must first import teneto
and define where the data is.

    >>> import teneto
    >>> import os
    >>> dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'

Here we see that there are two subjects.

    >>> os.listdir(dataset_path)
    ['participant.tsv', 'sub-001', 'derivatives', 'sub-002']

And in the derivatives we see two different derivatives that exist:

    >>> os.listdir(dataset_path + '/derivatives')
    ['teneto-tests', 'fmriprep']

Within the derivatives directory is where we find the preprocessed data.
We will use both of these two directories throughout the tutorial.
The fmriprep directory has some dummy data with confounds and fMRI-like data.

We can look at one of the subject:

    >>> os.listdir(dataset_path + '/derivatives/fmriprep/sub-001/func/')
    ['sub-001_task-a_run-02_bold_preproc.nii.gz',
    'sub-001_task-a_run-02_confounds.tsv',
    'sub-001_task-b_run-01_confounds.tsv',
    'sub-001_task-a_run-01_bold_preproc.nii.gz',
    'sub-001_task-b_run-01_bold_preproc.nii.gz',
    'sub-001_task-a_run-01_confounds.tsv']

Here we see there are two different tasks (a, b), and two different runs (01, 02).

There are both preprocessed nifti files (fMRI images)
and tsv files which contain confounds (ending with the suffix _confounds).

Within the confound files we have two confounds: "confound1" and "confound2"

In the teneto-tests directory there is also some dummy data time-series for regions of interest and time-varying connectivity. 

    >>> os.listdir(dataset_path + '/derivatives/teneto-tests/sub-001/func/')
    ['parcellation', 'tvc']

Here we see two directories, one contains the ROI data, one contains the TVC data. 
We call this directory in this tutorial for pipeline_subdir. 

Within these directories we find data that has a similar name as above.

Read More 
-------------

`BIDS specification page`_.

.. _a link: https://github.com/bids-standard/bids-specification