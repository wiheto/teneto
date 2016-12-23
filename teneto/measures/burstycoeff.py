import numpy as np
from teneto.utils import *
from teneto.measures.intercontacttimes import intercontacttimes
from functools import reduce

def burstycoeff(datIn,calcPer='edge',nodes='all'):
    """
    returns calculates the bursty coefficient. Value > 0 indicates bursty. Value < 0 periodic/tonic. Value = 0 implies random.
    As temporalPaths only works with binary undirected edges at the moment, weighted edges are assumed to be binary.

    Parameters
    ----------
    datIn: Variable input which can be either:
        A) Temporal graph of format (can be bu, bd):
            (i) G: graphlet (3D numpy array).
            (ii) C: contact (dictionary)
        B) Dictionary of ICTs (output of intercontacttimes function).
    calcPer: caclulate the bursty coeff for 'edge' (default), 'node' or 'meanEdgePerNode'.
        'edge': calculate B on all icts between node i and j.
        'node': caclulate B on all icts connected to node i.
        'meanEdgePerNode': first calculate the icts between node i and j, then take the mean over all j.
    nodes: which do to do.
        'all' (default) - all nodes
        List of indexes - do bursty between those nodes only.


    Returns
    ----------
    burst coefficienct per (edge or node measure)
    format: 1d numpy array

    See Also
    ----------
    intercontacttimes

    Origin of metric
    ----------
    Goh and Barabasi 2008
    Discrete formulation here from Holme 2012.

    History
    ----------
    Created - Nov 2016, WHT
    """

    ict=0 #are ict present
    if isinstance(datIn,dict):
        #This could be done better
        if [k for k in list(datIn.keys()) if k=='intercontacttimes']==['intercontacttimes']:
            ict=1
    # if shortest paths are not calculated, calculate them
    if ict==0:
        datIn = intercontacttimes(datIn)

    ictShape = datIn['intercontacttimes'].shape


    if len(ictShape)==2:
        l = ictShape[0]*ictShape[1]
    elif len(ictShape)==1:
        l = 1
    else:
        raise ValueError('more than two dimensions of intercontacttimes')

    if isinstance(nodes,list) and len(ictShape)>1:
        nodeCombinations=[[list(set(nodes))[t], list(set(nodes))[tt]] for t in range(0,len(nodes)) for tt in range(0,len(nodes)) if t!=tt]
        doNodes = [np.ravel_multi_index(n,ictShape) for n in nodeCombinations]
    else:
        doNodes = range(0,l)

    #Reshae ICTs
    if calcPer == 'node':
        ict = np.concatenate(datIn['intercontacttimes'][doNodex,doNodex],axis=1)

    if len(ictShape)>1:
        ict=datIn['intercontacttimes'].reshape(ictShape[0]*ictShape[1])
        B=np.zeros(len(ict))*np.nan
    else:
        B=np.zeros(1)*np.nan
        ict=[datIn['intercontacttimes']]

    for n in doNodes:
        mu = np.mean(ict[n])
        sigma = np.std(ict[n])
        B[n]=(sigma-mu)/(sigma+mu)
    if len(ictShape)>1:
        B=B.reshape(ictShape)
    return B
