Teneto and BIDS: TenetoBIDS
---------------------------

The BIDS format for neuroimaging is a way to organize data. If your data is organized this way, then teneto can use this to efficiently process the data.  

The aim of TenetoBIDS is a class of functions which run on preprocessed fMRI data and run some additional preprocessing steps, time-varying connectivity derivation and quantify temporal network properties.
The compatiblity with the BIDS format allows for Teneto to run the analysis on all subjects/files/tasks with only a few lines of code.  

.. toctree::
    :maxdepth: 1

    tenetobids/prerequisites
    tenetobids/data
    tenetobids/preprocess
    tenetobids/define
    tenetobids/derive
    tenetobids/io
    tenetobids/bidsinfo

