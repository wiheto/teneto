import numpy as np
import teneto
from teneto.misc import distance_functions
import collections

"""

Couple of utiltity functions for teneto for converting between graphlet and contact sequence representations

"""

def graphlet2contact(G,cfg=None):
    """

    Converts graphlet (sliced) representation of temporal network and converts it to contact representation representation of network. Contact representation are more efficient for memory storing. Also includes metadata which can made it easier for plotting. Contact representation can also theoretically be used to apply continuous time. But teneto does not support this yet. A contact is considered to be all non-zero edges.

    **PARAMETERS**

    :G: temporal network (graphlet)
    :cfg: config files for contact representation, a dictionary of meta information about the graph.
        Can be left empty and the function will try and assign everything necessary

        :Fs: sampling rate (number). default = 1.
        :timeunit: (string): Default = ''. What is the sampling rate in for units (e.g. seconds, minutes, years).
        :nettype: (string) can be:

            :'auto': (default) detects automatically.
            :'wd': weighted, directed
            :'bd': binary, directed
            :'wu': weighted, undirected
            :'bu': binary, undirected.

        :diagonal: (number). Default = 0. What should the diagonal be. (note: does could be expanded to take vector of unique diagonal values in the future, but not implemented now)
        :timetype: 'discrete' (only available option at the moment. But more may be added). The cfg file becomes the foundation of 'C'. Any other information in cfg, will added to C.
        :nLabs: node labels.
        :t0: time label at first index.


    **OUTPUT**

    :C: Contact representation of temporal network.

        :Format: Dictionary. Includes 'contacts', 'values' (if nettype[0]='w'),'nettype','netshape', 'Fs', 'dimord' and 'timeunit', 'timetype'.

    **NOTES**

    Until time permits to make code more efficient, many functions call contact2graphlet to make graphlets when calculating and this might not be ram efficient. This will be made better in later versions.


    **SEE ALSO**

    - *contact2graphlet*

    **HISTORY**

    :Modified: Dec 2016, WHT (documentaion, efficiency)
    :Created: Nov 2016, WHT

    """

    #Create config dictionary if missing
    if cfg==None:
        cfg={}
    #Check that temporal network is vald input.
    if G.shape[0]!=G.shape[1]:
        raise ValueError('Input G (node x node x time), requires Rows and Columns to be the same size.')
    if len(G.shape)==2:
        G=np.atleast_3d(G)
    if len(G.shape)!=3:
        raise ValueError('Input G must be three dimensions (node x node x time)')
    #Check number of nodes is correct, if specfied
    if 'nLabs' in cfg.keys():
        if len(cfg['nLabs']) != G.shape[0]:
            raise ValueError('Specified list of node names has to be equal in length to number of nodes')
    if 't0' in cfg.keys():
        cfg['t0']=np.atleast_1d(np.array(cfg['t0']))
        if len(cfg['t0'])!=1:
            raise ValueError('t0 must be sigular be either integer representing time at first temporal index)')
        cfg['t0']=np.squeeze(cfg['t0'])
    #Check that all inputs in cfg are correct.


    if 'nettype' not in cfg.keys() or cfg['nettype']=='auto':
        cfg['nettype'] = gen_nettype(G,1)
    if cfg['nettype'] not in {'bd','bu','wd','wu','auto'}:
        raise ValueError('\'nettype\' (in cfg) must be a string \'wd\',\'bd\',\'wu\',\'bu\'). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if 'Fs' not in cfg.keys():
        cfg['Fs']=1
        print('Warning, no sampling rate set. Assuming 1.')
    if 'timeunit' not in cfg.keys():
        cfg['timeunit'] = ''
    if 'diagonal' not in cfg.keys():
        cfg['diagonal'] = 0
    if 'nLabs' not in cfg.keys():
        cfg['nLabs'] = ''
    else:
        cfg['nLabs']=list(cfg['nLabs'])

    if 't0' not in cfg.keys():
        cfg['t0'] = 1
    nt=cfg['nettype']

    #Set diagonal to 0 to make contacts 0.
    G=set_diagonal(G,0)

    #Very convoluted way to get all the indexes into a tuple, ordered by time
    if nt[1]=='u':
        G=[np.triu(G[:,:,t],k=1) for t in range(0,G.shape[2])]
        G=np.transpose(G,[1,2,0])
    edg=np.where(np.abs(G)>0)
    sortTime = np.argsort(edg[2])
    contacts = np.array([tuple([edg[0][i],edg[1][i],edg[2][i]]) for i in sortTime])
    #Get each of the values if weighted matrix
    if nt[0]=='w':
        values = list(G[edg[0][sortTime],edg[1][sortTime],edg[2][sortTime]])


    #build output dictionary
    C = cfg
    C['contacts'] = contacts
    C['netshape'] = G.shape
    C['dimord'] = 'node,node,time'
    #Obviously this needs to change
    C['timetype'] = 'discrete'
    if nt[0]=='w':
        C['values'] = values

    return C





def contact2graphlet(C):
    """

    Converts contact representation to graphlet (sliced) representation.

    Graphlet representation discards all meta information.

    NOTE this is called automatically in many metric functions.

    **PARAMETERS**

    C: A contact representation.
        Must include 'dimord', 'netshape', 'nettype', 'contacts' and, if weighted, 'values'.

    **OUTPUT**

    :G: graphlet representation of temporal network.

        :format: 3 dimensional numpy array that is of the graph.

    **NOTES**

    Returning elements of G will be float, even if binary graph. Thus starting with G(integers) converting to C and then back to G with be float.

    **SEE ALSO**

    - *graphlet2contact*

    **HISTORY**

    :Modified: Dec 2016, WHT (documentation)
    :Created: Nov 2016, WHT

    """

    #Check that contact sequence is vald input.
    if 'dimord' not in C.keys():
        raise ValueError('\'dimord\' must be present in C.')
    if C['dimord']!='node,node,time':
        raise ValueError('\'dimord\' must be string \'node,node,time\'.')
    if 'dimord' not in C.keys():
        raise ValueError('\'dimord\' must be present in C.')
    if C['dimord']!='node,node,time':
        raise ValueError('\'dimord\' must be string \'node,node,time\'.')
    if 'nettype' not in C.keys():
        raise ValueError('C must include parameter \'nettype\' (wd,bd,wu,bu). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if C['nettype'] not in {'bd','bu','wd','wu'}:
        raise ValueError('\'nettype\' in (C) must be a string \'wd\',\'bd\',\'wu\',\'bu\'). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if 'netshape' not in C.keys():
        raise ValueError('C must include netshape expressing size of target network (tuple)')
    if not isinstance(C['netshape'],tuple):
        raise ValueError('\'netshape\' (in C) should be a tuple')
    if len(C['netshape'])!=3:
            raise ValueError('\'netshape\' tuple should be of 3 dimensions')
    if C['nettype'][0]=='w' and 'values' not in C.keys():
        raise ValueError('values not in C and asked for weighted network')
    if 'contacts' not in C.keys():
        raise ValueError('contacts must be expressed (list of tuples)')
    if C['timetype'] != 'discrete':
        print('Warning: timetype is not discrete. In future updates timetype in dictionary should be \'discrete\' to be converted to grpahlets')

    nt = C['nettype']

    #Preallocate
    G=np.zeros(C['netshape'])

    #Convert indexes of C to numpy friend idx list
    idx=np.array(list(map(list,C['contacts'])))
    if nt[0] == 'b':
        G[idx[:,0],idx[:,1],idx[:,2]]=1
        if nt[1] == 'u':
            G[idx[:,1],idx[:,0],idx[:,2]]=1
    elif nt[0] == 'w':
        G[idx[:,0],idx[:,1],idx[:,2]]=C['values']
        if nt[1] == 'u':
            G[idx[:,1],idx[:,0],idx[:,2]]=C['values']
    #If diagonal is not 0, fill it to whatever it is set to
    if C['diagonal'] != 0:
        G=set_diagonal(G,C['diagonal'])


    return G


def set_diagonal(G,val=0):
    """

    Generally diagonal is set to 0. This function helps set the diagonal across time.


    **PARAMETERS**

    :G: temporal network (graphlet)
    :val: value to set diagonal to (default 0).

    **OUTPUT**

    :G: Graphlet representation of G with new diagonal

    **HISTORY**

    :Modified: Dec 2016, WHT (documentation)
    :Created: Nov 2016, WHT

    """

    for t in range(0,G.shape[2]):
        np.fill_diagonal(G[:,:,t],val)
    return G


def gen_nettype(G,printWarning=0):
    """

    Attempts to identify what nettype input graphlet G is.
    Diagonal is ignored.

    **PARAMETERS**

    :G: temporal network (graphlet)
    :printWarning: 0 (default) or 1. Prints warning in console so user knows assumption

    **HISTORY**

    :Created: November 2016, WHT

    """

    if set(np.unique(G)) == set([0,1]):
        weights='b'
    else:
        weights='w'

    if np.allclose(G.transpose(1, 0, 2), G):
        direction='u'
    else:
        direction='d'

    nettype = weights + direction

    if printWarning == 1:
        netNames={'w':'weighted', 'b':'binary', 'u':'undirected','d':'directed'}
        print('Assuming network is ' + netNames[nettype[0]] + ' and ' + netNames[nettype[1]] + '.')

    return nettype


def checkInput(netIn,raiseIfU=1,conMat=0):
    """

    This function checks that netIn input is either graphlet (G) or contact (C).

    **PARAMETERS**

    :netIn: temporal network, either graphlet or contact representation.
    :raiseIfU: 1 (default) or 0. Error is raised if not found to be G or C
    :conMat: 0 (default) or 1. If 1, input is allowed to be a 2 dimensional connectivity matrix. Allows output to be 'M'

    **OUTPUT**

    :inputType: String indicating input type. 'G','C', 'M' or 'U' (unknown). M is special case only allowed when conMat=1 for a 2D connectivity matrix.

    **HISTORY**

    :Created: November 2016, WHT

    """

    inputIs='U'
    if isinstance(netIn,np.ndarray):
        netShape=netIn.shape
        if len(netShape)==3 and netShape[0]==netShape[1]:
            inputIs = 'G'
        if netShape[0]==netShape[1] and conMat == 1:
            inputIs = 'M'

    elif isinstance(netIn,dict):
        if 'nettype' in netIn and 'contacts' in netIn and 'dimord' in netIn  and 'timetype' in netIn:
            if netIn['nettype'] in {'bd','bu','wd','wu'} and netIn['timetype'] == 'discrete' and netIn['dimord'] == 'node,node,time':
                inputIs = 'C'

    elif isinstance(netIn,object):
        if isinstance(netIn.contact,dict):
            if 'nettype' in netIn.contact and 'contacts' in netIn.contact and 'dimord' in netIn.contact  and 'timetype' in netIn.contact:
                if netIn.contact['nettype'] in {'bd','bu','wd','wu'} and netIn.contact['timetype'] == 'discrete' and netIn.contact['dimord'] == 'node,node,time':
                    inputIs = 'TO'

    if raiseIfU == 1 and inputIs=='U':
        raise ValueError('Input cannot be identified as graphlet or contact representation')

    return inputIs

def getDistanceFunction(requested_metric):
    """

    This function returns a specified distance function.


    **PARAMETERS**

    :'requested_metric': can be 'hamming', 'eculidean'

    **OUTPUT**

    returns distance function (as function)

    **NOTE**

    New distance functions can be added in ./teneto/misc/distancefunctions.py and can be called when requested_metric = 'myDinstanceMetricName'

    **HISTORY**

    :Created: Dec 2016, WHT

    """

    if hasattr(distance_functions,requested_metric + '_distance'):
        return getattr(distance_functions,requested_metric + '_distance')
    else:
        raise ValueError('Distance function cannot be found. Check if your input distance funciton name is spelt correctly. Then check if it supported. If not supported you can add it in ./teneto/misc/distancefunctions.py or request that it gets added on github.')





def process_input(netIn,allowedformats,outputformat='G'):
    """
    Takes input, check what input is

    **PARAMETERS**

    :netIn: input network
    :allowedformats: list containing any of 'C', 'TO', 'G'
    :outputformat: target output format.

    **OUTPUT**

    :G: Graphlet representatoin
    :netInfo: Information about graphlet


    **HISTORY**

    *Created* - Jan17, WHT

    """
    inputType=teneto.utils.checkInput(netIn)
    #Convert TO to C representation
    if inputType == 'TO' and 'TO' in allowedformats:
        G = netIn.get_graphlet_representation()
        netInfo = dict(netIn.contact)
        netInfo.pop('contacts')
    #Convert C representation to G
    elif inputType == 'C' and 'C' in allowedformats and outputformat != 'C':
        G = teneto.utils.contact2graphlet(netIn)
        netInfo = dict(netIn)
        netInfo.pop('contacts')
        nettype = netIn['nettype']
    #Get network type if not set yet
    elif inputType == 'G' and 'G' in allowedformats:
        netInfo = {}
        netInfo['netshape'] = netIn.shape
        netInfo['nettype'] = teneto.utils.gen_nettype(netIn)
        G = netIn
    elif inputType == 'C' and outputformat == 'C':
        pass
    else:
        raise ValueError('Input invalid.')
    if inputType != 'C' and outputformat == 'C':
        C = teneto.utils.graphlet2contact(netIn,netInfo)
    if outputformat == 'G':
        return G,netInfo
    elif outputformat == 'C':
        return C


def clean_community_indexes(communityID):
    """
    Takes input of community assignments. Returns reindexed community assignment by using minimal amount of numbers.

    **PARAMETERS**

    :communityID: list or numpy array of integers.

    **OUTPUT**

    :new_communityID: cleaned list going from 0 to numberOfSubnetworks-1
        :format: numpy array of intergers
        :behaviour: the lowest community integer in communityID will recieve the lowest integer in new_communityID.

    **HISTORY**

    *Created* - Feb17, WHT

    """
    np.array(communityID)
    new_communityID = np.zeros(len(communityID))
    for i,n in enumerate(np.unique(communityID)):
        new_communityID[communityID==n]=i
    return new_communityID



def multiple_contacts_get_values(C):
    """
    Given an contact representation with repeated contacts, this function removes duplicates and creates a value

    **PARAMETERS**

    :C: contact representation with multiple repeated contacts.

    **OUTPUT**

    :C_out: contact representation with duplicates removed and the number of repeated contacts are now in the 'values' field.

    **HISTORY**

    *Created* - Feb17, WHT

    """
    d = collections.OrderedDict()
    for c in C['contacts']:
        ct = tuple(c)
        if ct in d:
            d[ct] += 1
        else:
            d[ct] = 1

    new_contacts = []
    new_values = []
    for (key, value) in d.items():
        new_values.append(value)
        new_contacts.append(key)
    C_out = C
    C_out['contacts']=new_contacts
    C_out['values']=new_values
    return C_out
