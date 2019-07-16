Teneto and BIDS: TenetoBIDS
---------------------------

The BIDS format for neuroimaging is a way to organize data.
If your data is organized this way,
then teneto can use this to efficiently process the data.

The aim of TenetoBIDS is use preprocessed fMRI data and run:

1. Additional preprocessing steps
2. Time-varying connectivity derivation
3. Quantify temporal network properties.

The compatibility with the BIDS format allows for Teneto
to run the analysis on all subjects/files/tasks with only a few lines of code.

.. toctree::
    :maxdepth: 1

    tenetobids/prerequisites
    tenetobids/data
    tenetobids/preprocess
    tenetobids/define
    tenetobids/derive
    tenetobids/networkmeasure
    tenetobids/io
    tenetobids/example
    tenetobids/bidsinfo

