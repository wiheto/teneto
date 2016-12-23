import numpy as np

"""

Couple of utiltity functions for teneto for converting between graphlet and contact sequence representations

"""

def graphlet2contact(G,cfg=None):
    """
    Converts graphlet (sliced) representation of temporal network and converts it to contact representation representation of network.

    Contact representation are more efficient for memory storing. Also includes metadata which can made it easier for plotting.

    Contact representation can also theoretically be used to apply continuous time. But teneto does not support this yet.

    A contact is considered to be all non-zero edges.

    Parameters
    ----------
    G: Temporal graph, directed or weighted, as numpy array (graphlet representation):
        (i) 3D numpy array of v x v x t (v=node d=time).
    cfg: config files for contact representation, a dictionary of meta information about the graph.
        Can be left empty and the function will try and assign everything necessary
        (i) Fs: sampling rate (number). default = 1.
        (ii) timeunit (string): Default = ''.
            What is the sampling rate in for units (e.g. seconds, minutes, years).
        (iii) nettype (string). Automatic detection to be added:
            'auto' (default) - detects automatically.
            'wd' - weighted, directed
            'bd' - binary, directed
            'wu' - weighted, undirected
            'bu' - binary, undirected.
        (iv) diagonal (number). Default = 0.
            What should the diagonal be.
            (note: does could be expanded to take vector of unique diagonal values in the future, but not implemented now)
        (v) timetype: 'discrete' (only available option at the moment. But more may be added)
        The cfg file becomes the foundation of 'C'. Any other information in cfg, will added to C.


    Returns
    ----------
    C, Contact representation of temporal network.
    Dictionary includes 'contacts', 'values' (if nettype[0]='w'),'nettype','netshape', 'Fs', 'dimord' and 'timeunit', 'timetype'.

    NOTES
    ----------
    Contact are more efficient for storing large sparse graphs to memory
    However, until time permits to make code more efficient, many functions call contact2graphlet to make graphlets when calculating and this might not be ram efficient. This will be made better in later versions.


    See Also
    ----------
    contact2graphlet

    History
    ----------
    Created, November 2016, WHT

    """

    #Create config dictionary if missing
    if cfg==None:
        cfg={}
    #Check that temporal network is vald input.
    if G.shape[0]!=G.shape[1]:
        raise ValueError('Input G (node x node x time), requires Rows and Columns to be the same size.')
    if len(G.shape)!=3:
        raise ValueError('Input G must be three dimensions (node x node x time)')

    #Check that all inputs in cfg are correct.

    if 'nettype' not in cfg.keys() or cfg['nettype']=='auto':
        cfg['nettype'] = gen_nettype(G,1)
    if cfg['nettype'] not in {'bd','bu','wd','wu','auto'}:
        raise ValueError('\'nettype\' (in cfg) must be a string \'wd\',\'bd\',\'wu\',\'bu\'). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if 'Fs' not in cfg.keys():
        cfg['Fs']=1
    if 'timeunits' not in cfg.keys():
        cfg['timeunites'] = ''
        print('Warning, no sampling rate set. Assuming 1.')
    if 'diagonal' not in cfg.keys():
        cfg['diagonal'] = 0

    nt=cfg['nettype']

    #Set diagonal to 0 to make contacts 0.
    G=set_diagonal(G,0)

    #Very convoluted way to get all the indexes into a tuple, ordered by time
    if nt[1]=='u':
        G=[np.triu(G[:,:,t],k=1) for t in range(0,G.shape[2])]
        G=np.transpose(G,[1,2,0])
    edg=np.where(np.abs(G)>0)
    sortTime = np.argsort(edg[2])
    contacts = [tuple([edg[0][i],edg[1][i],edg[2][i]]) for i in sortTime]
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

    Parameters
    ----------
    C: A contact representation.
        Must include 'dimord', 'netshape', 'nettype', 'contacts' and, if weighted, 'values'.

    Returns
    ----------
    3 dimensional numpy array that is of the graph.

    NOTES
    ----------
    Returning elements of G will be float, even if binary graph. Thus starting with G(integers) converting to C and then back to G with be float.


    See Also
    ----------
    graphlet2contact

    History
    ----------
    Created, November 2016, WHT

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
    if C['nettype'][0]=='w' and 'values 'not in C.keys():
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


    Parameters
    ----------
    G: Temporal graph, directed or weighted, as numpy array (graphlet representation):
        (i) 3D numpy array of v x v x t (v=node d=time).
    val: value to set diagonal to (default 0).

    Returns
    ----------
    Graphlet representation of G with new diagonal

    History
    ----------
    Created, November 2016, WHT

    """

    for t in range(0,G.shape[2]):
        np.fill_diagonal(G[:,:,t],val)
    return G


def gen_nettype(G,printWarning=0):
    """
    Attempts to identify what nettype input graphlet G is.
    Diagonal is ignored.

    Parameters
    ----------
    G: Temporal graph, directed or weighted, as numpy array (graphlet representation):
        (i) 3D numpy array of v x v x t (v=node d=time).
    printWarning = 0 (default) or 1. Prints warning in console so user knows assumption

    History
    ----------
    Created, November 2016, WHT

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

    Parameters
    -----------
    netIn = temporal network, either graphlet or contact representation.
    raiseIfU = 1 (default) or 0. Error is raised if not found to be G or C
    conMat = 0 (default) or 1. If 1, input is allowed to be a 2 dimensional connectivity matrix.

    Returns
    -----------
    String indicating input type.
    'G','C' or 'U' (unknown)

    History
    ----------
    Created, November 2016, WHT

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

    if raiseIfU == 1 and inputIs=='U':
        raise ValueError('Input cannot be identified as graphlet or contact representation')

    return inputIs

def getDistanceFunction(requested_metric):
    """
    This function returns a specified distance function.


    Parameters
    -----------
    'requested_metric' - can be 'hamming', 'eculidean', 'taxicab'

    NOTE
    ----------
    New distance functions can be added in ./teneto/misc/distancefunctions.py and can be called when requested_metric = 'myDinstanceMetricName'

    Returns
    -----------
    Distance function

    History
    ----------
    Created, December 2016, WHT

    """
    from teneto.misc import distancefunctions as df
    if hasattr(df,requested_metric + '_distance'):
        return getattr(df,requested_metric + '_distance')
    else:
        raise ValueError('Distance function cannot be found. Check if your input distance funciton name is spelt correctly. Then check if it supported. If not supported you can add it in ./teneto/misc/distancefunctions.py or request that it gets added on github.')
