
Workflows
--------------------------

Many analyses can be constructed as a graph to depict all the steps that are made during the analysis.
 This graph of an analysis is called a workflow.
 There are many benefits to creating a workflow:

- Construct entire analysis workflow and view it before running.
- Carefully records every step, so you know exactly what you did.
- Can share the entire analysis with someone else (good for reproducibility).

TenetoWorkflow allows you to define a workflow object and then run it.
 A workflow consists of a directed graph.
 The nodes of this graph are different Teneto functions.
 The directed edges of the graph is the sequence the pipeline is run in.

The workflows function around the TenetoBIDS or TemporalNetwork classes.
 Any analysis made using those classes can be made into a workflow.

There are three different types of nodes in this graph:

*root nodes*: These are nodes that do not depend on any other step in the analysis.
 These are calls to create a _TenetoBIDS_ or _TemporalNetwork_ object.

*non-terminal nodes*:
 These are nodes that are intermediate steps in the anlaysis.

*terminal nodes*:
 These are the final nodes in the analysis. 
 These nodes will include the output of the analysis.

Understanding the concept of root and terminal nodes are useful
 to understand how the input and output of TenetoWorkflow.

Creating a workflow
====================

We are going to create a workflow that does the following three steps:

1. Creates a temporal network object (root node)
2. Generates random data (non-terminal node)
3. Calculates the temporal degree centrality of each node (terminal node)

We start by creating a workflow object, and defining the first node:

    >>> from teneto import TenetoWorkflow
    >>> twf = TenetoWorkflow()
    >>> nodename = 'create_temporal_network'
    >>> func = 'TemporalNetwork'
    >>> twf.add_node(nodename=nodename, func=func)

Each node in the workflow graph needs a unique name (argument: nodename).
If you create two different TemporalNetwork objects in the workflow
, these need different names to differentiate them.

The func argument specifies the class that is initiated or
 the function that is run.

There are two more optional arguments that can be passed to add_node:
 depends_on and params. We will look at those later though.

By adding a node,
 this creates an attribute in the workflow object which can be viewed as:

    >>> twf.nodes
    {'create_temporal_network': {'func': 'TemporalNetwork', 'params': {}}}

It also creates a graph (pandas dataframe)
 which is found in TenetoWorkflow.graph.

    >>> twf.graph
        i   j
    0   isroot  create_temporal_network

Since this is the first node in the workflow,
 _isroot_ is placed in the _i_ column
 to signify that _create_temporal_network_ is the root node.

Now let us add the next two nodes and we will see the params argument add_node:

    >>> # Generate network node
    >>> nodename = 'generate_network'
    >>> func = 'generate_network'
    >>> params = {
        'networktype': 'binomial',
        'size': (10,5),
        'prob': (0.5,0.25)
        }
    >>> twf.add_node(nodename, func, params=params)
    >>> # Calc temporal degree centrality node
    >>> nodename = 'degree'
    >>> func = 'calc_networkmeasure'
    >>> params = {
        'networkmeasure': 'temporal_degree_centrality'
        }
    >>> twf.add_node(nodename, func, params=params)

Here we see that the params argument is a dictionary of _*kwargs_
 for the _TemporalNetwork.generate_network_
 and _TemporalNetwork.calc_networkmeasure_ functions.

Now we have three nodes defined,
 so we can look at the TenetoWorkflow.graph:

    >>> twf.graph
        i   j
    0   isroot  create_temporal_network
    1   create_temporal_network generate_network
    2   generate_network    degree

Each row here shows the new node in the _j_-th column
 and the step preceeding node in the _i_-th column.

The workflow graph can be plotted with:

    >>> fig, ax = twf.make_workflow_figure()
    >>> fig.show()

.. plot::

    from teneto import TenetoWorkflow
    twf = TenetoWorkflow()
    nodename = 'create_temporal_network'
    func = 'TemporalNetwork'
    twf.add_node(nodename=nodename, func=func)
    # Generate network node
    nodename = 'generate_network'
    func = 'generate_network'
    params = {
        'networktype': 'binomial',
        'size': (10,5),
        'prob': (0.5,0.25)
        }
    twf.add_node(nodename, func, params=params)
    # Calc temporal degree centrality node
    nodename = 'degree'
    func = 'calc_networkmeasure'
    params = {
        'networkmeasure': 'temporal_degree_centrality'
        }
    twf.add_node(nodename, func, params=params)
    fig, ax = twf.make_workflow_figure()
    fig.show()

