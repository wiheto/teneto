Load/Save TenetoBIDS
==========================

The configurations of a TenetoBIDS object can be
easily saved by using *save_tenetobids_snapshot*.

.. code-block:: python

    path = './'
    tnet.save_tenetobids_snapshot(path)

This creates a json file called, _TenetoBIDS_snapshot.json_
(unless you set the filename argument).

Then to load it you just need to create a new TenetoBIDS
and pass the json file as a dictionary.

.. code-block:: python

    import json
    with open(path + 'TenetoBIDS_snapshot.json') as f
        params = json.load(f)
    tnet = teneto.TenetoBIDS(**params)

All the history (in tnet.history) is preserved when saving.
However, any loaded files need to be reloaded.
