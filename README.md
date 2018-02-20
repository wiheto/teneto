# Teneto

**Te**mporal **ne**twork **to**ols. Version 0.2.5

Package includes:  temporal network measures, temporal network generation, derivation of time-varying/dynamic connectivity, plotting functions.

For neuroimaging data, compatible with BIDS format.

## Installation

With pip installed:

`pip install teneto`

to upgrade teneto:

`pip install teneto -U`

Requires: Python 3.5 or python 3.6 and pip.

Package dependencies:
- matplotlib>=1.5.3
- nilearn>=0.2.6
- numpy>=1.13.1
- pandas>=0.20.3
- scipy>=0.18.1
- statsmodels>=0.8.0

Other versions of python (2.7 and <3.5) may work, but not tested.

### Tutorials

__These will be updated__

[1. The different formats of temporal networks in teneto](https://github.com/wiheto/teneto/blob/master/examples/01_network_representations.ipynb)


[2. The plotting different networks](https://github.com/wiheto/teneto/blob/master/examples/02_plotting_temporalnetworks.ipynb)


[3. Creating temporal networks ](https://github.com/wiheto/teneto/blob/master/examples/03_creating_temporalnetworks.ipynb)

*More tutorials will be written when I have time (and the above will be improved). Until then, check out our overview of the metrics here: http://biorxiv.org/content/early/2016/12/24/096461*

### Documentation

Documentation for the functions can be found at  [teneto.readthedocs.io](https://teneto.readthedocs.io),  At some point the tutorials and documentation will merge.


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
- intercontacttimes

All measures work for binary undirected networks. Some work for other types of networks as well. This will be updated with time.

Found a measure in the literature that you would like included? Add to this issue with a reference to someone using it and I will try and implement it.

### Outlook.

This package is under active development. And a lot of changes will still be made.

## Cite

If using this, please cite us. At present we do not have a dedicated article about teneto, but teneto is introduced, a long with a considerable amount of the metrics:

Thompson, William Hedley, Per Brantefors, and Peter Fransson. "From static to temporal network theory applications to functional brain connectivity." Network Neuroscience (2017).
