# Teneto

**Te**mporal **ne**twork **to*ols. 

By William Hedley Thompson (wiheto)

This is still a lot of work to go before this package is complete. However, it is built to be used now and should not break people's pipelines. 

The first functions have now been uploaded. Some tutorial examples will be uploaded in the coming weeks, as too will network generation tools. 

## Installation

Clone this package and add to your python path.

On the to do list is to make it pip-able for easier install. But this will be done once the package is a little more mature.  

Requirements:

- numpy
- matplotlib
- python3.x

## Data formats (graphlets and contact).

The two data formats of temporal networks. These are called *grahlet* representation or *contact* representation after what the temporal network literature refers to them as. In practice, this is a numpy or dictionary format.

These formats can be converted between with `graphlet2contact()` and `contact2graphlet()` functions.

### Graphlet representation

The graphlet representation is just a 3D numpy array with the dimension order: node,node,time.

Advantages: easy to work with, query. Disadvantages: takes up a lot of memory when large.

### Contact representation

The contact representation is a python dictionary with each non-zero edge expressed as (i,j,t). The name comes from contact sequences which generally report (i,j,(t_start,t_end)). At the moment teneto does not have the sequence part and each temporal edge must be expressed as a contact. The dictionary includes:

- *contacts*: indexes of tuples expressing where and when a non-zero edge is. Tuple order is specified in dimord (but should always be node,node,time for now).
- *values*: (only if non-binary network), list of weights. value[x] corresponds to the contact[x] tuple.
- *netshape*: size of overall network. This is needed, especially if there is no edge present at the final time point.
- *nettype*: either 'bu', 'wu', 'bd', 'wd' where w=weighted, b=binary, u=undirected, d=directed.
-  *dimord*: dimensional order (should be node,node,time for now. But later updates may include more complex dimord).  
- *diagonal*: diagonal is also removed from contacts (for space reasons). Diagonal is generally treated as 0. (Note, currently only one value, and not a vector for all self edges is possible. Doesn't feel like high priority, but if requested can be implemented.)
- *timeunit*: default is ''. But will be used in plotting labeling functions if specified.
- *timetype*: 'discrete'. Only discrete metrics are included at the moment. We may someday expand to continuous time (but this will require a lot of work) and timetype is here so that nothing breaks if I implement this. Also, for even more memoruy efficiency, I will make 'discreteseq' when I have time which is (i,j,(t_start,t_end)) for discrete time only.
- *Fs*: sampling rate. Default = 1. If timeunit = 'seconds' and each time-point represents a 1 millisecond, then Fs should be 1000.  

Any additional information can be added to the contact representation and is kept. However, if you convert to graphlet form, all this information is lost.

Advantages: memory efficient, allows for more meta-information about graphs.

Disadvantages: perhaps less intuitive.  

Note: while the contact  is a more efficient way to store larger network data. A lot of the measures currently operate with graphlet representation *within functions*. The conversion from contact to graphlet is done within the functions, so nothing to worry about. But this can take up a more RAM for large graphs (as both contact and graphlet representations will exist when function is running). This will be removed when possible (but give me time).

### Measures

The following measures exist in the package:

- temporal degree
- temporal closeness
- shortest temporal paths
- bursty coefficient
- fluctuability  
- volatility
- reachability latency
- temporal efficiency

All measures work for binary undirected networks. Some work for other types of networks as well. This will be updated with time.

Found a measure in the literature that you would like included? Add to this issue with a reference to someone using it and I will try and implement it.

### Plotting

While note primarily a plotting package, there are some plotting tools included.

Plotting networks in time is hard. Plotting connectivity matrices in static cases can be counter intuitive.

### Outlook.

This package is under active development.

(1) Calculate based on contact sequence representations
(2) Assist in creating temporal network representations.
(3) integrate with python neuroimaging toolboxes.
(4) Make a better package structure (i.e. not imports in every function), place on pip, generate full documentation.

The time line is that all of these should be up over the next 6 months or so. If something is missing that you need now, let me know and I can try and prioritize getting that up and running (if requests are limited!).


## Cite

If using this, please cite us. At present we do not have a dedicated article about teneto, but teneto is introduced, a long with a considerable amount of the metrics:

Thompson, WH, Brantefors, P, Fransson, P. From static to temporal network theory – applications to functional brain connectivity. http://biorxiv.org/content/early/2016/12/23/096461.article-metrics
