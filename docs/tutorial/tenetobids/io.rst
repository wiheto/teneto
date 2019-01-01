Load saved data
=================

You can save you progress by using save_pickle 

.. code-block:: python

    tnet.save_aspickle(dataset_path + '/tenetoobj.pkl')

Then to load it you just need to write

.. code-block:: python

    tnet = teneto.TenetoBIDS.load_frompickle(dataset_path + '/tenetoobj.pkl')
