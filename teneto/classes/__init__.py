"""Classes in Teneto"""
#from teneto.classes.network import NewTemporalNetwork
from .bids import TenetoBIDS
from .network import TemporalNetwork
from .workflow import TenetoWorkflow
__all__ = ['TenetoBIDS', 'TemporalNetwork', 'TenetoWorkflow']
#from teneto.classes.preproc import fMRIPreproc
