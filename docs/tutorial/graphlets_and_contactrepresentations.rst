Network representation in Teneto
--------------------------------

There are three ways that network's are represented in Teneto:

1. A TemporalNetwork object
2. Numpy array/graphlet/snapshot
3. Dictionary/contact representation

This tutorial goes through what these different representations are and how to translate between them.

TemporalNetwork object
=======================

TemporalNetwork is a class in teneto. 

  >>> from teneto import TemporalNetwork
  >>> tnet = TemporalNetwork()
  ... 

As an input, you can pass it a 3D numpy array, a contact representation (see below), a list of edges or a pandas df. 

The nice feature of the TemporalNetwork class is that the different plotting and networkmeasures can be accessed within the object. 

  >>> tnet.generatenetwork('rand_binomial',size=(5,3), prob=0.5)

This calls the function *teneto.generatenetwork.rand_binomial* with all subsequent argument sbeing arguments for the *rand_binomial* function

The data this creates is found in *tnet.network* which is a pandas dataframe. To have a peak at the top of the datafram, we can call: 

  >>> tnet.network.head()
     i  j  t
  0  0  1  0
  1  0  1  1
  2  0  2  2
  3  0  3  0
  4  0  4  2

Each line in the dataframe represents one edge. *i* and *j* are both node indexes and *t* is a temporal index.  
There is no *weight* column here which indicates this is a binary network (all edges are one). We can see this is a binary network by calling: 

  >>> tnet.nettype
  'bu'

There are 4 different nettypes: bu, wu, wd and bd where b is for binary, w is for weighted, u means undirected and d means directed. 

To get a weighted network 

  >>> import numpy as np 
  >>> np.random.seed(2019)
  >>> G = np.random.beta(1, 1, [5,5,3]) # Creates 5 nodes and 3 time-points
  >>> tnet = TemporalNetwork(from_array=G, nettype='wd', diagonal=True)
  >>> tnet.network.head()
       i    j    t    weight
  0  0.0  1.0  0.0  0.856509
  1  0.0  1.0  1.0  0.518670
  2  0.0  1.0  2.0  0.370951
  3  0.0  2.0  0.0  0.673422
  4  0.0  2.0  1.0  0.539778

Self edges get deleted unless the argument *diagonal=True* is passed. Thet nettype should
be specfied whenever known (otherwise it can accidently get assumed as undirected). 

You can export the network back to a numpy array using.  

  >>> G2 = tnet.to_array()
  >>> G == G2
  True

Array/graphlet/snapshot representation
================================

A graphlet/snapshort representation is a three dimensional numpy array. The dimensions are (node,node,time). 

The positives of this representation is that it is easy to understand and manipulate. The downside is that any metainformation about the network is lost. 


Contact representation
================================

The contact representations is a dictionary that includes more information about the network. 

The keys in the dictionary include 'contact' which specified the network information (node,node,timestamp). A weights key is present in weighted networks containing the weights. 
Other keys include: 'dimord' (dimension order), 'Fs' (sampling rate), 'timeunit', 'nettype' (if network is weighted/binary, undirected/directed), 'timetype', `nLabs` (node labels), `t0` (the first time point). 

Converting between contact and graphlet representations
======================================================

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

