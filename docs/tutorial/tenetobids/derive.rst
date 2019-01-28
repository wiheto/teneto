Deriving time-varying representations 
======================================

First let's import what we need: 

    >>> import teneto
    >>> import numpy as np 
    >>> import matplotlib as plt 
    >>> dataset_path = teneto.__path__[0] + '/data/testdata/dummybids/'

To derive the time varying connectivity estimates, there are several available options in Teneto: 

- Sliding window 
- Tapered sliding window 
- Jackknife correlatoin 
- Spatial distance correlation 
- Multiply temporal derivatives 

Researchers have different preferences for different methods.

In TenetoBIDS, you can call the derive function and it will calculate the TVC estimates for you and placed
them in a tvc directory. 

In this example, we start out by selecting some dummy ROI data which is prespecfied in the teneto-tests directory. 

    >>> pipeline='teneto-tests'
    >>> data_directory = 'parcellation'
    >>> bids_tags = {'sub': '001', 'task': 'a', 'run': 'alpha'}
    >>> tnet = teneto.TenetoBIDS(dataset_path, pipeline=pipeline, pipeline_subdir=data_directory, bids_suffix='roi', bids_tags=bids_tags, raw_data_exists=False)

This contains 2 time-series which are 20 timepoints long. To see what we are working with, we can load the parcellation data and plot it. 

    >>> tnet.load_data('parcellation')
    >>> tnet.parcellation_data_[0]
    >>> fig,ax = plt.subplots(1)
    >>> ax.plot(np.arange(1,21),tnet.parcellation_data_[0].transpose())
    >>> ax.set_ylabel('Amplitude')
    >>> ax.set_xlabel('Time')
    >>> plt.tight_layout()
    >>> fig.show() 

.. plot::

    import matplotlib.pyplot as plt 
    import numpy as np
    # This is hardcoding the output from the functions in the document
    parcellation_data = np.array([  [-0.00856245, -0.72492072],
                                    [-1.32189023, -1.70531216],
                                    [-0.09987815,  0.53725969],
                                    [-0.18838177, -0.19577621],
                                    [-0.69506541, -0.21947204],
                                    [ 1.57793574,  1.54734448],
                                    [-0.99980587,  0.07231908],
                                    [ 0.06589326,  0.27025436],
                                    [ 0.40749625,  0.40945534],
                                    [ 0.4617551 , -0.02803148],
                                    [ 0.47971502,  0.98430683],
                                    [ 1.53515526,  1.19368654],
                                    [-0.07981899,  0.48576559],
                                    [ 0.29510053,  0.73294405],
                                    [-2.47835632, -0.53347075],
                                    [ 1.3517407 ,  1.3420725 ],
                                    [ 2.27989294,  0.62249916],
                                    [ 1.48691269,  0.05008697],
                                    [-0.43029765, -0.16380152],
                                    [-0.11351284,  1.24652865]])
    fig,ax = plt.subplots(1)
    ax.plot(np.arange(1,21),parcellation_data)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time')
    ax.set_xticks(np.arange(5,21,5))
    plt.tight_layout()
    fig.show() 

Let us say we want to apply the jackknife correlation method to this. To do this we just need to specify a dictionary of parameters which goes into teneto.timeseries.derive_temporalnetwork.
In the example below, we simply are saying we would to use the jackknife method and afterwards these estimates should be standerdized. 

    >>> derive_params = {'method': 'jackknife', 'postpro': 'standardize'}
    >>> tnet.derive_temporalnetwork(derive_params, confound_corr_report=False)
    ...

Setting *confound_corr_report* to true places a HTML showing histograms of each time-series each of the confounds so you can see how much the TVC is effected by them.

Now we have the time-varying estimates for each time-point, we can load and them by: 

    >>> tnet.load_data('tvc')

This produces a list of dataframes in *tnet.tvc_data\_*. 

    >>> tnet.tvc_data_[0].head()
         i    j    t    weight
    0  0.0  1.0  0.0 -0.829939
    1  0.0  1.0  1.0  1.830899
    2  0.0  1.0  2.0 -0.278181
    3  0.0  1.0  3.0  0.108855
    4  0.0  1.0  4.0  0.417800

Where we see the columns for nodes (i,j), time-points (t) and the connectivity estimate (weight). 

These lists of connectivity estimates are for space purposes. They can be conveted to an array format (node,node,time) by 
calling *teneto.TemporalNetwork* (this may be included within TenetoBIDS at a later release): 

    >>> tvc = teneto.TemporalNetwork(from_df=tnet.tvc_data_[0])
    >>> conn_time_series = tvc.to_graphlet() 
    >>> conn_time_series
    (2, 2, 20)

Now as an array, we can easily visualise the connectivity time series between the two nodes. 

    >>> fig,ax = plt.subplots(1)
    >>> ax.plot(np.arange(1,21),conn_time_series[0,1,:])
    >>> ax.set_ylabel('Connectivity estimate (Jackknife)')
    >>> ax.set_xlabel('Time')
    >>> plt.tight_layout()
    >>> fig.show()     

.. plot::

    import matplotlib.pyplot as plt 
    import numpy as np
    # This is hardcoding the output from the functions in the document
    conn_time_series = np.array([-0.82993863,  1.83089895, -0.27818135,  0.10885456,  0.41779984,
        1.7061645 , -0.2095942 ,  0.00961699,  0.03107385, -0.40607624,
       -0.07326018,  1.20480326, -0.19810589, -0.09228326,  0.54444428,
        1.15963977, -1.23889445, -2.02336494,  0.26725356, -1.93085042])
    fig,ax = plt.subplots(1)
    ax.plot(np.arange(1,21),conn_time_series)
    ax.set_ylabel('Connectivity estimate (Jackknife)')
    ax.set_xlabel('Time')
    ax.set_xticks(np.arange(5,21,5))
    plt.tight_layout()
    fig.show()     
