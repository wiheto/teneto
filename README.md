# Teneto

[![Documentation Status](https://readthedocs.org/projects/teneto/badge/?version=latest)](http://teneto.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/Teneto.svg)](https://badge.fury.io/py/Teneto)

**Te**mporal **ne**twork **to**ols. Version 0.3.0

## What is the package

Package includes various tools for analyzing temporal network data. Temporal network measures, temporal network generation, derivation of time-varying/dynamic connectivity, plotting functions. Some extra focus is placed on neuroimaging data (e.g. compatible with BIDS data format).

## Installation

With pip installed:

`pip install teneto`

to upgrade teneto:

`pip install teneto -U`

Requires: Python 3.5 or python 3.6. Other versions of python (2.7 and <3.5) may work, but not tested.

Installing teneto installs all python package requirements as well. However, for community detection, iGraph is used which can need some seperate C compilers installed. See ([python-iGraph installation page](http://igraph.org/python/#startpy) for more details regarding what needs to be installed).

##

Documentation for the functions can be found at  [teneto.readthedocs.io](https://teneto.readthedocs.io) and includes tutorials.

## Outlook.

This package is under active development. And a lot of changes will still be made.

## Cite

If using this, please cite us. At present we do not have a dedicated article about teneto, but teneto is introduced, a long with a considerable amount of the metrics:

Thompson, William Hedley, Per Brantefors, and Peter Fransson. "From static to temporal network theory applications to functional brain connectivity." Network Neuroscience (2017).
