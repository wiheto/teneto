What is temporal network theory?
=================================

This page goes over some of the basic concepts of temporal network theory and concepts. 

Node and edges: the basics of a network 
---------------------------------

A network is a representation. Usually it is a representation of some empirical phenomena but it can also be abstract (such as a simulation).
The representation of a network contains nodes (sometimes called vertices) and edges (sometimes called links).

Nodes and edges can represent a vast amount of different things in the world. For example, nodes can be friends, cities, or brain regions and their 
representative edges could be trust relationships, train lines, and neuronal communication. 

The benefits of network representation is that similar analysis methods can be applied, regardless of what the underlying node or edge represents. 
This means that network theory is a very inter-disciplinary subject (this however means sometimes things get multiple names). 

With a network, you can analyse for example, if there is any "hub" node. 
In transportation networks, there are often hubs which connect many different areas where passengers often have to change at (e.g. airports like Frankfurt, Heathrow or Denver)
In social networks you can quantify how many steps it is to another person in that network (see the famous 6 steps to Kevin Bacon)

Mathematically, A network if often referenced as G or mathcal(G). i and j reference nodes. A tuple (i,j) references an edge between nodes i and j. G is often 
expressed in the form of a connectivity matrix (or adjacency matrix) A_{ij} = 1 if a connection is present and A_{ij} 0 if a connection is not present. The number of nodes if often referenced to as N. 
Thus, A is a NxN matrix.  

Different network types
-----------------------

THere are a few different versions of networks. Two key properties distinctions are:

1. Are the connections *binary* or *weighted*. 
2. Are the connections *undirected* or *directed*. 

If a connection is binary, then (as in the section above) an edge is either present or not. When a weight is added, an edge is now represented as a 3-tuple (i,j,w) where w is the magnitude of the weight. 
And in the connectivity matrix, A_{ij} = w. Often the weight is between 0 and 1 or -1 and 1, but this does not have to be the case. 

When connections are undirected, it means that both nodes share the connection. Examples of such networks can be if two cities are connected by train lines. For such networks A_{ij} = A_{ji}. 
When connections are directed, it means that the connection goes from i to j. Examples of these types of networks can be citation networks. 
If a scientific article i cites another article j, it is not common for j to also cite i. So in such cases, A_{ij} does not need to equal A_{ji}. 
It is common notation for the source node (sending the information) to be written first and the target node (receiving the information) to be second.   

Adding a time dimension
-----------------------

In the above formulation of networks, there A_{ij} only has one edge. In a temporal network, a time-stampe is also applied for an edge. 
Thus, binary edges are not expressed as 3-tuples (i,j,t) and weighted networks as 4 tuples (i,j,t,w). 
Connectivity matrices are now three dimensional: A_{ijt} = 1 in binary and A_{ijt} = w in weighted networks.
The time indices are an ordered sequence. This can have a consequence about how what has happening in the network and reveal information about what is occurring in the network.

For example, using friends' lists from social network profiles can be used to create a static network about who is friends with who. 
However, imagine a friend being introduced to a group of friends, by seeing when they become friends this can explain more what happened. 

Compare the following two figures representing meetings between friends: 

.. plot::

    import matplotlib.pyplot as plt 
    import numpy as np
    import teneto 
    G = np.zeros([5,5,4])
    G[0,1,0] = 1
    G[2,3,1] = 1
    G[0,3,1] = 1
    G[1,2,1] = 1
    G[0,3,1] = 1
    G[1,4,2] = 1
    G[0,4,3] = 1
    G[3,4,3] = 1
    fig, ax = plt.subplots(1,2)
    teneto.plot.slice_plot(G, ax=ax[1], cmap='Set2', timeunit='Event', nodelabels=['Ashley', 'Blake', 'Casey', 'Dylan', 'Elliot'])
    ax[1].set_title('Temporal network')
    G2 = G.sum(axis=-1)
    G2[G2>0] = 1
    teneto.plot.circle_plot(G2, ax=ax[0])
    ax[0].set_title('Static network')
    fig.tight_layout() 
    fig.show()

In the static network, on the left, each person (node) is a circle and each black line connecting the circles is an edge. 
In this figure, we we can see that everyone has met everyone except Dylan (orange) and Casey (green). 

The slice_plot on the left shows nodes (circles) at multiple "slices" (time-points). Each column represent of nodes represents one time-point. The black line connecting two nodes at a time-point
 signifies that they met at that time-point. 

In the temporal network, we can see a progression of who met who and when. At event 1, Ashley and Blake met. Then A-D all met together at event 2. At event 3, Blake met Dylan. 
And at event 4, Elliot met Dylan and Ashley (but those two themselves did not meet). This allows for new properties to be quantified that are simply missed in the static network.


What is time-varying connectivity? 
-----------------------------------

Another concept that is often used within cognitive neuroscience is time-varying connectivity. 

Time-varying connectivity is a larger domain of methods. 

*More to come here*

What is teneto?
-----------------

Teneto is a python package that can several quantify temporal network measures (more are always being added). 
It can also used methods from time-varying connectivity to derive connectivity estimate from time series data. 

Further reading
---------------