TenetoBIDS & Temporal network measure
========================

Once a time-varying connectivity representation has been derived, different temporal network measures can be derived. 

Which network measure should be used? 
------------------------------------

While there can be explorative studies which test multiple measures without any hypohthesis, it is often better to have a 
hypothesis first. Network measures quantify an abstract representation (the network). With a hypothesis about how you think the underlying 
concrete phenomena is, you can select the most appropriate network measure, and then it is easier to interpret what the abstract network measures represents. 

Say for example, you think there are periods of high integration between two brain networks, then there are a couple of methods 
that help indicate this (e.g. , _SID_, _temporal_ _degree_ _centrality_). Using one of those will help you translate 
to that hypothesis. But if you instead find a difference in _volatility_, without considering what it represents in your use-case, 
it can be harder to translate back to a concrete interpretation. It is often best to start out with "what do I think is happening regarding the temporal properties in the brain" and 
then see which measure best captures such actions than properties to identify abstract network representation.

Network measures in TenetoBIDS
-------------------------------

In TenetoBids, if there is tvc representation in the teneto derivatives directory, you only need to call the calc_networkmeasures attribute.

    >>> import teneto 
    >>> dataset_path = teneto.__path__[0] + '/data/testdata/dummybids'
    >>> pipeline = 'teneto-tests'
    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline=pipeline, raw_data_exists=False)

Now let us select the ROI data for subject 2, task b, run alpha:

    >>> tnet.set_pipeline_subdir('parcellation')
    >>> bids_tags = {'sub': '002', 'task': 'b', 'run': 'alpha'}
    >>> tnet.set_bids_tags(bids_tags)
    >>> tnet.bids_suffix = 'roi'

Check that we have selected the right file:

    >>> tnet.get_selected_files(quiet=1)[0]
    '.../derivatives/teneto-tests/sub-002/func/parcellation/sub-002_task-b_run-alpha_roi.tsv'

Which is correct. 

Let us load this data to see what it looks like: 

    >>> tnet.load_data('parcellation')
    >>> fig, ax = plt.subplots(1)
    >>> ax.plot(tnet.parcellation_data_)
    >>> ax.set_xlabel('Time')
    >>> ax.set_xticks(np.arange(49,200,50))
    >>> ax.set_xticklabels(np.arange(50,201,50))
    >>> fig.show() 

.. plot:: 
    import numpy as np 
    import matplotlib.pyplot as plt
    import pandas as pd
    import teneto  
    # hodecode data 
    dataset_path = teneto.__path__[0] + '/data/testdata/dummybids'
    data_path = '/derivatives/teneto-tests/sub-002/func/parcellation/sub-002_task-b_run-alpha_roi.tsv'
    data = pd.read_csv(dataset_path + data_path, sep='\t', index_col=[0])
    fig, ax = plt.subplots(1)
    ax.plot(data.transpose())
    ax.set_xlabel('Time')
    ax.set_xticks(np.arange(49,200,50))
    ax.set_xticklabels(np.arange(50,201,50))
    fig.show() 

We can see here that there are three time series with a bit of noise. The blue and the orange both follow sin curves that are offset from eachother. 
The green time series switchest which sin curve it follows. This illustrate some-type of switching beahviour connectivity. 

We can calculate the TVC data from this

    >>> tnet.derive({'method': 'jackknife', 'postpro': 'standardize'})

We can see the jackknife estimates for the relationship between the Green and Orange time series above. 

    tnet.load_data('tvc')
    # Note, there will be an easier function to do this soon. 
    # But the below creates an array from the loaded pandas dictionary 
    G = teneto.TemporalNetwork(from_df=tnet.tvc_data_[0]).to_array()
    fig, ax = plt.subplots(1)
    ax.plot(G[1,2,:], color='green')


.. plot:: 
    import numpy as np 
    import matplotlib.pyplot as plt
    import pandas as pd
    import teneto  
    # hodecode data 
    dataset_path = teneto.__path__[0] + '/data/testdata/dummybids'
    data_path = '/derivatives/teneto-tests/sub-002/func/tvc/sub-002_task-b_run-alpha_tvcconn.tsv'
    data = pd.read_csv(dataset_path + data_path, sep='\t', index_col=[0])
    G = teneto.TemporalNetwork(from_df=data).to_array()
    fig, ax = plt.subplots(1)
    ax.plot(G[1,2,:],color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Jackknife estimate')
    ax.set_xticks(np.arange(49,200,50))
    ax.set_xticklabels(np.arange(50,201,50))
    fig.show() 

We can then use this measure to calculate the volatility of the network (i.e. rate of change). 

    >>> tnet.networkmeasures('volatility')

This then saves a new directory called: temporalnetwork-volatility with the file inside. To load this, simply type

    >>> tnet.load_data('temporalnetwork',measure='volatility')

The data is then found in the temporalnetwork_data_ argument. This is a dictionary, where the key is the name of the measure you loaded. 

    >>>  tnet.temporalnetwork_data_['volatility']
    [   volatility
     0     0.93812]

Here we see that the volatility of the network above is 0.938.