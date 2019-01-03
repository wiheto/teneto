Defining TenetoBIDS
===================

First let's import what we need: 

    >>> import teneto
    >>> dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'

The first step is to create a TenetoBIDS object by calling the TenetoBIDS class. There are (at leas) two arguments 
that are needed, the BIDS_dir (i.e. the base directory) and the pipeline (the folder in the derivatives folder where the data is). 

    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='fmriprep', raw_data_exists=False)

Usually the argument raw_data_exists does not need to be passed, the reason for passing it is to ignore the data outside of the derivatives directories which is useful for how we have made this dummy data. But usually this can be ignored. 

Certain arguments can be passed when calling the TenetoBIDS class. 
For example, the data consists of tasks, runs, and subjects. To choose specific ones of these we can pass a dictionary of BIDS tags: 

    >>> bids_tags = {'sub': '001', 'task': 'a', 'run': 'alpha'}
    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='fmriprep',  bids_tags=bids_tags, raw_data_exists=False)

This will mean that only subject \'001\', task \'a\' and \'run\' alpha are loaded. It is also possible to set bids tags after you define the teneto object. 

    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='fmriprep', raw_data_exists=False)
    >>> tnet.set_bids_tags({'sub': '001'})
    >>> tnet.set_bids_tags({'task': 'a', 'run': 'alpha'})

To see which files tnet identified within the BIDS directory and provide a summary of which files are currently selected, type: 

.. code-block:: python

    tnet.print_dataset_summary()

Which will print something like this: 

|    --- DATASET INFORMATION ---
|    --- Subjects ---
|    Number of subjects (selected): 1
|    Subjects (selected): 001
|    Bad subjects: 0
|    --- Tasks ---
|    Number of tasks (selected): 1
|    Tasks (selected): a
|    --- Runs ---
|    Number of runs (selected): 1
|    Rubs (selected): alpha
|    --- Sessions ---
|    Number of sessions (selected): 0
|    Sessions (selected): 
|    --- PREPROCESSED DATA (Pipelines/Derivatives) ---
|    Pipeline: fmriprep
|    --- SELECTED DATA ---
|    Numnber of selected files: 1
|    [list of files]

This helps provide some information about the data. 

If you ever want to retrive data from the BIDS structure, there are two that can be used. 
tnet.load_data() and tnet.get_selected_files(). The former will be demonstrated later. 

    >>> paths = tnet.get_selected_files(quiet=1)

Will produce a list of paths for the files that have been selected. The quiet argument states if this information is printed in the terminal or not. 

Whenever data is manipulated, it gets copied to a new derivatives directory called teneto\_[tenetonumber]. This updates your pipeline.

    >>> bids_tags = {'sub': '001', 'task': 'a', 'run': 'alpha'}
    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='fmriprep',  bids_tags=bids_tags, raw_data_exists=False)
    >>> tnet.derive({'method': 'jackknife'})
    >>> tnet.pipeline
    teneto...

An important argument used in selection is called pipeline_subdir. If for some reaso files are not being found, it is probably because this is not specified or missspecified. 
In the directory [bidsdir]/derivatives/teneto\_[versionnumber]/sub-[xxx]/[ses-[xxx]]/func/[pipeline_subdir]/. This helps sort the data. (Note: it is unclear if this is BIDS compliant and may be removed.) 

    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='teneto-tests', raw_data_exists=False, pipeline_subdir='tvc')

This makes sure you select the tvc data. 

BIDS data also has different types of suffixes to identify differnet types of data. For example \'_preproc\' signifies preprocessed data. This argument can be passed to make sure 
you only select a certain typue of data. For example: 

    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline='teneto-tests', raw_data_exists=False, pipeline_subdir='tvc', bids_suffix='tvcconn')



