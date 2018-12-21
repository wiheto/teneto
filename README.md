# Teneto

[![Documentation Status](https://readthedocs.org/projects/teneto/badge/?version=latest)](http://teneto.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/Teneto.svg)](https://badge.fury.io/py/Teneto)
[![Build Status](https://travis-ci.org/wiheto/teneto.svg?branch=master)](https://travis-ci.org/wiheto/teneto)
[![Coverage Status](https://coveralls.io/repos/github/wiheto/teneto/badge.svg?branch=master)](https://coveralls.io/github/wiheto/teneto?branch=master)

**Te**mporal **ne**twork **to**ols. Version 0.4.0 (development)

## What is the package

Package includes various tools for analyzing temporal network data. Temporal network measures, temporal network generation, derivation of time-varying/dynamic connectivity, plotting functions. Some extra focus is placed on neuroimaging data (e.g. compatible with BIDS - _NB: currently not compliant with latest release candidate of BIDS Derivatives_).

## Installation

With pip installed:

`pip install teneto`

to upgrade teneto:

`pip install teneto -U`

Requires: Python 3.5 or python 3.6. 

Installing teneto via pip installs all python package requirements as well. 

## Installation notes

Version 0.3.5+: community detection has been temporarily removed until a better solution than using iGraph is found. iGraph has lead to multiple problems on non-linux systems. Community detection can still be imported (import teneto.communitydeteciton) but it has been removed from TenetoBIDS. If required in TenetoBIDS the code still exists but is commented uncomment ./teneto/classes/bids.py line 1060-1132 to get working again.  

## Documentation

More detailed documentation can be found at  [teneto.readthedocs.io](https://teneto.readthedocs.io) and includes tutorials.

## Outlook

This package is under active development. And a lot of changes will still be made.

## Cite

If using this, please cite us. At present we do not have a dedicated article about teneto, but teneto is introduced, a long with a considerable amount of the metrics in:

Thompson et al (2017) "From static to temporal network theory applications to functional brain connectivity." *Network Neuroscience*, 2: 1. p.69-99  [Link](https://www.mitpressjournals.org/doi/abs/10.1162/NETN_a_00011)
