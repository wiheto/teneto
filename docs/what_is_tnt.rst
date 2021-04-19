What are temporal networks?
=================================

Temporal networks are, quite simply, network representations that flow through time. They are useful for analysing how a connected system develops, changes or evolves through time. This change in time can depict how information spreads along with a social network or how different brain areas cooperate to perform a task.

This page introduces some of the basic concepts of temporal network theory.

Node and edges: the basics of a network
------------------------------------------

A network is a representation of *something* using a graph from mathematics.
This *something* can be a representation of an empirical phenomenon or a simulation.
A graph contains nodes (sometimes called vertices) and edges (sometimes called links).

Nodes and edges can represent a vast amount of different things in the world. For example, nodes can be friends, cities, or brain regions. Edges between could represent trust relationships, train lines, and neuronal communication.

The flexibility in what nodes are is one of the reasons network theory is very interdisciplinary. The benefits of having network representation are that similar analysis methods can be applied, regardless of what the underlying node or edge represents. This abstractness means that network theory is a very inter-disciplinary subject. However, it also entails that certain concepts have multiple names (e.g. nodes and vertices).

With a network, you can analyse for example, if there is any "hub" node. In transportation networks, there are often hubs which connect many different areas where passengers usually have to change at (e.g. airports like Frankfurt, Heathrow or Denver). In social networks, you can quantify how many steps it is to another person in that network (see the famous six steps to Kevin Bacon).

Mathematically, A network if often referenced as G or :math:`\mathcal(G)`; *i* and *j* are indices of nodes; a tuple *(i,j)* reference an edge between nodes *i* and *j*. *G* is often expressed in the form of a connectivity matrix (or adjacency matrix) :math:`A_{ij} = 1` if a connection is present and :math:`A_{ij} = 0` if a connection is not present. The number of nodes if often referenced to as *N*. Thus, *A* is a *N x N* matrix.

Different network types
-----------------------

There are a few different versions of networks. Two key distinctions are:

1. Are the connections *binary* or *weighted*.
2. Are the connections *undirected* or *directed*.

If a connection is binary, then (as in the section above) an edge is either present or not. When adding a weight-value, an edge becomes a 3-tuple *(i,j,w)* where *w* is the magnitude of the weight. And in the connectivity matrix, :math:`A_{ij} = w`. Often the weight is between 0 and 1 or -1 and 1, but this does not have to be the case.

When connections are undirected, it means that both nodes share the connection. Examples of such networks can be if two cities are connected by train lines. For such networks :math:`A_{ij} = A_{ji}`. With directed edges, it means that the connection goes from *i* to *j*. Examples of these types of networks can be citation networks. If a scientific article *i* cites another article j, it is not common for *j* to also cite *i*. So in such cases, :math:`A_{ij}` does not need to equal :math:`A_{ji}`. It is the common notation for the source node (sending the information) to be first and the target node (receiving the information) to be second.

Adding a time dimension
-----------------------

In the above formulation of networks :math:`A_{ij}` only has one edge. In a temporal network, a time-stamp is also included in the edge's tuple. Thus, binary edges are now expressed as 3-tuples (i,j,t) and weighted networks as 4 tuples (i,j,t,w). Connectivity matrices are now three dimensional: :math:`A_{ijt} = 1` in binary and :math:`A_{ijt} = w` in weighted networks.

The time indices are an ordered sequence. This ordering can now reveal information about what is occurring in the network through time.

For example, using friends' lists from social network profiles can be used to create a static network about who is friends with who.
However, imagine one person enters a group of friends, by seeing when everyone become friends, this gives the network more explanatory power.

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

In the static network, on the left, each person (node) is a circle, and each black line connecting the rings is an edge. In this figure, we can see that everyone has met everyone except Dylan (orange) and Casey (light green).

The slice_plot on the left shows nodes (circles) at multiple "slices" (time-points). Each column represents of nodes represents one time-point. The black line connecting two nodes at a time-point signifies that they met at that time-point.

In the temporal network, we can see a progression of who met who and when. At event 1, Ashley and Blake met. Then A-D all met together at event 2. At event 3, Blake met Dylan. And at event 4, Elliot met Dylan and Ashley (but those two themselves did not attend). This depiction allows for new properties to be quantified that missed in a static network.


What is time-varying connectivity?
-----------------------------------

Another concept that is often used within fields such as cognitive neuroscience is *time-varying connectivity*. Time-varying connectivity is a larger domain of methods that analyse distributed patterns over time where temporal network theory is one set of analysis methods within it. Temporal network theory analyses time-varying connectivity representations that consist of time-stamped edges between nodes. There are other alternatives to analyse such representations and other time-varying connectivity representations as well (e.g. temporal ICA).

What is teneto?
-----------------

Teneto is a python package that can several quantify temporal network measures (more are being added). It can also use methods from time-varying connectivity to derive connectivity estimate from time-series data.

Further reading
---------------

Holme, P., & Saramäki, J. (2012). Temporal networks. Physics reports, 519(3), 97-125. [`Arxiv link <https://arxiv.org/pdf/1108.1780.pdf>`_] - Comprehensive introduction about core concepts of temporal networks.

Kivelä, M., Arenas, A., Barthelemy, M., Gleeson, J. P., Moreno, Y., & Porter, M. A. (2014). Multilayer networks. Journal of complex networks, 2(3), 203-271. [`Link <https://academic.oup.com/comnet/article/2/3/203/2841130>`_] - General overview of multilayer networks.

Lurie, D., Kessler, D., Bassett, D., Betzel, R. F., Breakspear, M., Keilholz, S., ... & Calhoun, V. (2018). On the nature of resting fMRI and time-varying functional connectivity. [`Psyarxiv link <https://psyarxiv.com/xtzre/download?format=pdf>`_] - Review of time-varying connectivity in human neuroimaging.

Masuda, N., & Lambiotte, R. (2016). A Guidance to Temporal Networks. [`Link to book's publisher <https://www.worldscientific.com/doi/abs/10.1142/9781786341150_0001>`_] - Book that covers a lot of the mathematics of temporal networks.

Nicosia, V., Tang, J., Mascolo, C., Musolesi, M., Russo, G., & Latora, V. (2013). Graph metrics for temporal networks. In Temporal networks (pp. 15-40). Springer, Berlin, Heidelberg. [`Arxiv link <https://arxiv.org/pdf/1306.0493>`_] - Review of some temporal network metrics.

Thompson, W. H., Brantefors, P., & Fransson, P. (2017). From static to temporal network theory: Applications to functional brain connectivity. Network Neuroscience, 1(2), 69-99. [`Link <https://www.mitpressjournals.org/doi/full/10.1162/netn_a_00011>`_] - Article introducing temporal network's in cognitive neuroscience context.
