Network representation in Teneto
--------------------------------

There are two ways that network's are represented in Teneto:

1. As a graphlet/snapshot
2. Contact representation

This tutorial goes through what these different representations are and how to translate between them.

Graphlet/snapshot representation
================================

A graphlet/snapshort representation is a three dimensional numpy array. The dimensions are (node,node,time). 

The positives of this representation is that it is easy to understand and manipulate. The downside is that any metainformation about the network is lost. 

Contact representation
================================

The contact representations is a dictionary that includes more information about the network. 

The keys in the dictionary include 'contact' which specified the network information (node,node,timestamp). A weights key is present in weighted networks containing the weights. 
Other keys include: 'dimord' (dimension order), 'Fs' (sampling rate), 'timeunit', 'nettype' (if network is weighted/binary, undirected/directed), 'timetype', `nLabs` (node labels), `t0` (the first time point). 

Converting between representations
==================================

Converting between the two different network representations is quite easy. First let us generate a random network that consists of 3 nodes and 5 time points. 

.. code-block:: python

  import teneto
  import numpy as np

  # For reproduceability
  np.random.seed(2018) 
  # Number of nodes
  N = 3
  # Number of timepoints
  T = 5
  # Probability of edge activation
  p0to1 = 0.2
  p1to1 = .9
  G = teneto.generatenetwork.rand_binomial([N,N,T],[p0to1, p1to1],'graphlet','bu')
  # Show shape of network
  print(G.shape)
    
You can convert a graphlet representatoin to contact representation with teneto.utils.graphlet2contact

.. code-block:: python

  C = teneto.utils.graphlet2contact(G)
  print(C.keys)

To convert the opposite direction, type teneto.utils.contact2graphlet and check that the new numpy array is equal to the previous one. 

.. code-block:: python
  G2 = teneto.utils.contact2graphlet(C)
  G==G2
