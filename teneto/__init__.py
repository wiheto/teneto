
# #!/usr/bin/env python3
#
"""Teneto is a module with tools for analyzing network patterns in time."""
#
__author__ = "William Hedley Thompson (wiheto)"
__version__ = "0.4.0"
#
from . import networkmeasures
from . import utils
from . import plot
from . import generatenetwork
from . import derive
from . import misc
from . import trajectory
#from . import communitydetection
from .classes import TenetoBIDS, TemporalNetwork
#del misc.teneto
#del communitydetection.static.modularity_based_clustering

