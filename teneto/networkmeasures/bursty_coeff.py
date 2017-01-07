import numpy as np
from teneto.utils import *
from teneto.networkmeasures.intercontacttimes import intercontacttimes
from functools import reduce

def bursty_coeff(datIn,calcPer='edge',nodes='all'):
    """
    returns calculates the bursty coefficient. Value > 0 indicates bursty. Value < 0 periodic/tonic. Value = 0 implies random.
    As temporalPaths only works with binary undirected edges at the moment, weighted edges are assumed to be binary.

    **PARAMETERS**

    :datIn: This is either:

        :netIn: temporal network input (graphlet or contact).

            :nettype: 'bu', 'bd'

        :ICT: dictionary of ICTs (output of *intercontacttimes*).

    :calcPer: caclulate the bursty coeff over what. Options include

        :'edge': calculate B on all ICTs between node i and j. (Default)
        :'node': caclulate B on all ICTs connected to node i.
        :'meanEdgePerNode': first calculate the ICTs between node i and j, then take the mean over all j.

    :nodes: which do to do. Options include:

        :'all': do for all nodes (default)
        :specify: list of node indexes to calculate.


    **OUTPUT**

    :B: bursty coefficienct per (edge or node measure)

        :format: 1d numpy array

    **SEE ALSO**

    intercontacttimes

    **ORIGIN**

    Goh and Barabasi 2008
    Discrete formulation here from Holme 2012.

    **HISTORY**

    :Modified: Nov 2016, WHT (documentation)
    :Created: Nov 2016, WHT

    """

    ict=0 #are ict present
    if isinstance(datIn,dict):
        #This could be done better
        if [k for k in list(datIn.keys()) if k=='intercontacttimes']==['intercontacttimes']:
            ict=1
    # if shortest paths are not calculated, calculate them
    if ict==0:
        datIn =     intercontacttimes(datIn)

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
