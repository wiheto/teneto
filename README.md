# Teneto

**Te**mporal **ne**twork **to**ols. Version 0.1.4 (Feb 2, 2016)

This is still a lot of work to go before this package is complete. However, it is built to be used now and future updates *should* not break people's pipelines.

Package includes: temporal network generation (only one function at the moment), temporal network measures, plotting functions.

## Installation

Requires: Python 3.x and pip.

Package dependencies:
- numpy
- scipy
- matplotlib

### Installation

At the moment the easiest way to install is the following:

1. Download the package manually [here](https://github.com/wiheto/teneto/archive/master.zip) or clone by `git clone  https://github.com/wiheto/teneto.git`
2. With pip install by `pip install /path/to/teneto/`

If a previous versions of Teneto has been installed, download the new version and  `pip install /path/to/teneto/ -U`

### Tutorials

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

All measures work for binary undirected networks. Some work for other types of networks as well. This will be updated with time.

Found a measure in the literature that you would like included? Add to this issue with a reference to someone using it and I will try and implement it.

### Outlook.

This package is under active development.

(1) Calculate based on contact sequence representations
(2) Assist in creating temporal network representations.
(3) integrate with python neuroimaging toolboxes and other network formats from other packages.

The time line is that all of these should be up over the next 6 months or so. If something is missing that you need now, let me know and I can try and prioritize getting that up and running (if requests are limited!).


## Cite

If using this, please cite us. At present we do not have a dedicated article about teneto, but teneto is introduced, a long with a considerable amount of the metrics:

Thompson, WH, Brantefors, P, Fransson, P. From static to temporal network theory – applications to functional brain connectivity. http://biorxiv.org/content/early/2016/12/24/096461
