Load/Save TenetoBIDS
=================

TenetoBIDS objects can be easily saved as a pickle file (.pkl). 

You can save you progress by using *save_pickle*.

.. code-block:: python

    tnet.save_aspickle('./tenetoobj.pkl')

Then to load it you just need to call *load_frompickle*:

.. code-block:: python

    tnet = teneto.TenetoBIDS.load_frompickle('./tenetoobj.pkl')
