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
