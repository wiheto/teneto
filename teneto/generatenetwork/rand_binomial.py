import numpy as np
from teneto.utils import graphlet2contact
def rand_binomial(size,p,netrep='graphlet',nettype='bu',initialize='zero',netinfo=None):

    """

    Creates a random binary network following a binomial distribution.

    **PARAMETERS**

    :size: number of nodes and number of time points.

        :format: 2-tuple, list of size 2 or array of size 2. Can also be of length 3 (node x node x time) but number of nodes in 3-tuple must be identical.

    :p: Probability of edge present. Two possibilities.

        :integer: the same probabability for each node becoming active (equal for all nodes).
        :tuple/list of size 2: different probabilities for edges to become active/inactive.

            The first value is "birth rate". The probability of an absent connection becoming present.

            The second value is the "death rate". This dictates the probability of an edge present remaining present.

            :example: (40,60) means there is a 40% chance that a 0 will become a 1 and a 60% chance that a 1 stays a 1.

    :netrep: network representation: 'graphlet' or 'contact'.
    :nettype: string 'bu' or 'bd' (accepts 'u' and 'd' as well as b is implicit)
    :initialize: optional variable for option2 of p. Follwoing options:

        :'zero': all nodes start deactivated
        :integer: states percentage of nodes that should active when t=1 (determined randomly).
    :netinfo: dictionary for contact representaiton information


    **OUTPUT**

    :net: generated network.

        :format: either graphlet (numpy) or contact (dictionary), depending on netrep input.

    **NOTES**

    Option 2 of the p parameter can be used to create a small autocorrelaiton or make sure that, once an edge has been present, it never disapears.

    **SEE ALSO**

    **READ MORE**
    There is some work on the properties on the graphs with birth/death rates (called edge-Markovian Dynamic graphs) as described here. Clementi et al (2008) Flooding Time in edge-Markovian Dynamic Graphs *PODC*

    **HISTORY**

    :Created: Dec 16, WHT

    """

    size=np.atleast_1d(size)
    p = np.atleast_1d(p)
    if len(size)==2 or (len(size==3) and size[0]==size[1]):
        ok=1
    else:
        raise ValueError('size input should be [numberOfNodes,Time]')
    if len(p)>2:
        raise ValueError('input: p must be of len 1 or len 2')
    if p.min()<0 or p.max()>1:
        raise ValueError('input: p should be probability between 0 and 1')
    if nettype[-1] == 'u' or nettype[-1] == 'd':
        ok = 1
    else:
        raise ValueError('nettype must be u or d')

    N = size[0]
    T = size[-1]
    cm = N*N
    if len(p)==1:
        net=np.random.binomial(1,p,cm*T)
        net=net.reshape(N*N,T)
    if len(p)==2:
        net=np.zeros([cm,T])
        if initialize=='zero':
            t_start=0
        else:
            edgesAt0=np.random.randint(0,cm,int(np.round(initialize*(cm))))
            net[edgesAt0,0]=1
        for t in range(0,T-1):
            e0 = np.where(net[:,t]==0)[0]
            e1 = np.where(net[:,t]==1)[0]
            ue0=np.random.binomial(1,p[0],len(e0))
            ue1=np.random.binomial(1,p[1],len(e1))
            net[e0,t+1]=ue0
            net[e1,t+1]=ue1
    #Set diagonal to 0
    net[np.arange(0,N*N,N+1),:]=0
    #Reshape to graphlet
    net=net.reshape([N,N,T])
    #only keep upper left if nettype = u
    #Note this could be made more efficient by only doing (N*N/2-N) nodes in cm and inserted directly into upper triangular.
    if nettype[-1]=='u':
        unet=np.zeros(net.shape)
        ind=np.triu_indices(N)
        unet[ind[0],ind[1],:] = np.array(net[ind[0],ind[1],:])
        unet = unet + np.transpose(unet,[1,0,2])
        net = unet
    if netrep == 'contact':
        if netinfo == None:
            netinfo={}
        netinfo['nettype'] = 'b' + nettype[-1]
        net=graphlet2contact(net,netinfo)
    return net
