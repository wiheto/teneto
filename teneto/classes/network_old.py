import numpy as np
import teneto.networkmeasures
import teneto.utils
import copy

def add_node(netObj,n=1,nLab=''):
    """
    Adds a node to the end of network.
        :n: How many nodes to add. Default 1. Can be any integer.
        :nLab: list of labels for added nodes. len(nLab)==n
    """
    #If input is string make to list. Repeat list if n>1 and nLab is string.
    if nLab != '' and isinstance(nLab,str):
        nLab = [nLab] * n

    #Check that number of nLabs is compatible with number of new nodes. Add new nLabs
    if nLab!='' and n!=len(nLab):
        raise ValueError('Specified node labels need to each number of new nodes')
    elif netObj.contact['nLabs']!='' and nLab == '':
        netObj.contact['nLabs'] = netObj.contact['nLabs'] + ['UnnamedNode'] * n
    elif nLab != '':
        netObj.contact['nLabs'] = netObj.contact['nLabs'] + nLab

    #Add new
    netObj.contact['netshape']=tuple([netObj.contact['netshape'][0]+n, netObj.contact['netshape'][1]+n, netObj.contact['netshape'][2]])
    return netObj


def add_time(netObj,t=1):
    """
    Adds a time-point to the end of network.
        :t: How many time-points to add. Default 1. Can be any integer.
    """
    netObj.contact['netshape']=tuple([netObj.contact['netshape'][0], netObj.contact['netshape'][1], netObj.contact['netshape'][2]+t])
    return netObj

def add_edge(netObj,e,dimRefNode='index'):
    """
    Adds a an edge.

    :e: nx3 (bu,bd) or nx4 (wu,wd)array. Where n is the number of edges to be added
        :for binary: columns are [node,node,time]
        :for wegithed: columns are [node,node,time,value]

    :dimRefNode: reference node 'index' or 'nLabs'
    """
    e=np.array(e)
    if netObj.contact['nettype'][0]=='b' and e.shape[-1]!=3:
        raise ValueError('For binary networks, e must have len of 3 (node,node,time)')
    elif netObj.contact['nettype'][0]=='w' and e.shape[-1]!=4:
        raise ValueError('For weighted networks, e must have len of 4 (node,node,time,value)')
    for edge in e:
        if edge[0] >= netObj.contact['netshape'][0] or edge[1] >= netObj.contact['netshape'][1]:
            raise ValueError('Nodes must be defined before nodes passes through edge')
        if edge[2] >= netObj.contact['netshape'][2]:
            raise ValueError('Time index must be defind before edge defind')
        if np.sum(np.sum(netObj.contact['contacts']-edge==0,axis=1)==3)>0:
            raise ValueError('Edge already exists')
        netObj.contact['contacts']=np.vstack([netObj.contact['contacts'],edge[0:3]])
        if netObj.contact['nettype'][0]=='w':
            netObj.contact['values']=np.vstack([netObj.contact['values'],edge[3]])
    return netObj

def add_graphlet(netObj,G=[]):
    """
    Adds a graphlet.

    :G: 2 or 3 dimensional graphlet.

    """
    if G==[]:
        netobj = add_time(netObj)
    elif netObj.contact['netshape'][0]!=G.shape[0] or netObj.contact['netshape'][1]!=G.shape[1]:
        raise ValueError('Graphlets added must be of same size')
    else:
        C = teneto.utils.graphlet2contact(G)
        netObj = add_contact(netObj,C)
    return netObj

def add_contact(netObj,C,time='append'):
    """
    Adds a graphlet.

    :C: contact of the same size.
    :time: 'append' means all contacts are added to the end. 'asis', contacts are added as they are.

    """
    if netObj.contact['netshape'][0]!=C['netshape'][0]:
        raise ValueError('Contact shapes added must be of same size')
    if time == 'append':
        C = copy.deepcopy(C)
        C['contacts'][:,-1] += netObj.contact['netshape'][-1]
        netObj.contact['netshape']=(netObj.contact['netshape'][0],netObj.contact['netshape'][1],netObj.contact['netshape'][-1]+C['netshape'][-1])
        netObj.contact['contacts']=np.array(np.vstack([netObj.contact['contacts'],C['contacts']]))

    return netObj


class NewTemporalNetwork:

    def __init__(self):

        self.contact = []


    def generate_rand_binomial(self,size,p,nettype='bu',initialize='zero',netinfo=None):
        if netinfo != None:
            netinfo = dict(netinfo)
        self.contact= teneto.generatenetwork.rand_binomial(size,p,'contact',nettype,initialize,netinfo)


    def get_shape(self,):
        print(self.contact['netshape'])

    def get_nettype(self,):
        print(self.contact['nettype'])

    def get_graphlet_representation(self,):
        return teneto.utils.contact2graphlet(self.contact)

    def from_graphlet_representation(self,G):
        self.contact = teneto.utils.graphlet2contact(G)

    def CreateHyperGraph(self):
        return teneto.HyperGraph(self)

    def add_node(self,n=1,nLab=''):
        self = add_node(self,n,nLab)

    def add_time(self,t=1):
        self = add_time(self,t)

    def add_edge(self,e,dimRefNode='index'):
        self = add_edge(self,e)

    def add_graphlet(self,G=[]):
        self = add_graphlet(self,G)

    def add_contact(self,C,time='append'):
        self = add_contact(self,C,time=time)

# class HyperGraph:
#
#     def __init__(self):
#
#         self.contact = []
#
#     def get_shape(self,):
#         print(self.contact['netshape'])
#
#     def get_nettype(self,):
#         print(self.contact['nettype'])
#
#     def get_graphlet_representation(self,):
#         return teneto.utils.contact2graphlet(self.contact)
#
#     def from_graphlet_representation(self,G):
#         self.contact = teneto.utils.graphlet2contact(G)
#
#     def add_node(self,n=1,nLab=''):
#         self = add_node(self,n,nLab)
#
#     def add_edge(self,e,dimRefNode='index'):
#         self = add_edge(self,e)
