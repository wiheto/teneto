
Burstiness Coefficient 
---------------------

Conceptual Background
=====================

The connections within a temporal network can behave with certain patterns. 
Sometimes an edge is always present, sometimes it is connected with a periodic time period (e.g. every 10 seconds). 
Sometimes an edges is random. At other times it may appear with a "bursty" pattern. 
Burstiness is a pattern which is found in many temporal networks (@Barabasi2005). 
For example, they are found in email contact networks. 
Given a specific edge, it is very often that, if active, several emails will be sent within a short period of time. 
However, once the "burst" is complete there is a varying intermittent period before the next period. 
This activity is caracterized by a fat tailed distribution. 

In order to quantify the amount of burstiness a given distribution of _intercontact_ _times_ (i.e. collection of time periods between connections). 
The equation for this was presented @Goh2008 and @Holme2012, which is: 

B = (mean(tau) - std(tau)) / (mean(tau) - std(tau))    

The value of B must be between -1 and 1. If B = 0, then the pattern of intercontact times is random. If B < 0, it implies a periodic distribution of intercontacttimes. 
If B > 0 it is indicative of a bursty pattern. 

Bursty_Coeff
============

To illustrate B, let's start of by creating a network with three nodes. The edges between node 1 and node 2 will be random (following poisson distribution). The edges between 1 and 3 will be periodic. And finally, the edges between 2 and 3 will be bursty
First import numpy and teneto 

.. code-block:: python

  import teneto
  import numpy as np

.. code-block:: python

  G = np.zeros([3,3,10000])
  # Add poisson distribution (make this better) 
  lam = 1
  i = 0 
  while i < G.shape[-1]: 
      i += np.random.poisson(lam)
      if i >= G.shape[-1]: 
          break
      else:
          G[0,1,i] = 1
  # Add periodic 
  G[0,2,np.arange(0,G.shape[-1],2)] = 1 
  # Add exponential  (make this better)
  ict = np.round(np.random.exponential(10,G.shape[-1]))
  i = 0 
  j = 0
  while i < G.shape[-1]: 
      i += int(ict[j])
      if i >= G.shape[-1]: 
          break
      else:
          G[1,2,i] = 1 
      j += 1

With the above network definition it is posisble to calculate the bursty coefficent quite easily. 

.. code-block:: python

  B = teneto.networkmeasures.bursty_coeff(G)

Here we see that B is: 

Calculating intercontact times
============================== 

If interestied in bursty processes, you may be interested in more than just the burstiness coefficient. It is possible to calculate the intercontact times for each edge. 

.. code-block:: python

  icts = teneto.networkmeasures.intercontacttimes(G)

icts is a dicitonary. Within this dictionary, there is icts['intercontacttimes'] which is a node,node matrix. Each field in the matrix consists of an array of intercontact times for that edge.  
So icts['intercontacttimes'][0,1] will show the icts for the edge between node 0 and node 1. 