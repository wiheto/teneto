BIDS suffixes used in tenetoBIDS 
========================

The following suffixes can get applied to files while using the TenetoBIDS functions. 

+-----------+-------------------------------------+
| Suffix    | Description                         |
+===========+=====================================+
| _tvcconn  | time-varying connectivity estimate  |
+-----------+-------------------------------------+
| _conn     | functional connectivity estimate    |
+-----------+-------------------------------------+
| _roi      | region of interst time series       |
+-----------+-------------------------------------+
| _tnet     | temporal network estimate           |
+-----------+-------------------------------------+

File formats are all .tsv files. 

Note: These are not approved in the BIDS derivaives/connectivity at the moment as the connectivity is not finalized yet. So these may change over time.
For example, the current suggestion is to use _connectivity as a suffix. 


Information in JSON sidecars
----------------------------

The JSON sidecars offer information about the different files. 
All the input parameters for functions are listed there. 
There is still some work needed on this aspect of the data. 


