Load/Save TenetoBIDS
=================

TenetoBIDS objects can be easily saved as a pickle file (.pkl). 

You can save you progress by using *save_tenetobids_snapshot*.

.. code-block:: python

    path = './'
    tnet.save_tenetobids_snapshot(path)

This creates a json file called, 
Then to load it you just need to create a new TenetoBIDS object using these paramerers. 
All the history (in tnet.history) is also perserved:

.. code-block:: python

    import json
    with open(path + 'TenetoBIDS_snapshot.json') as f
        params = json.load(f)
    tnet = teneto.TenetoBIDS(**params)

Note that any loaded files need to be reloaded.