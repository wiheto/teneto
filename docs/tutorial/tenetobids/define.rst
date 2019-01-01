Defining TenetoBIDS
===================

First let's import what we need: 

    >>> import teneto
    >>> dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'

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
