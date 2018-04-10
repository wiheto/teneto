import numpy as np
import teneto
import collections
import scipy.spatial.distance as distance
from nilearn.input_data import NiftiSpheresMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
import json

"""

Couple of utiltity functions for teneto for converting between graphlet and contact sequence representations

"""


def graphlet2contact(G, params=None):
    """

    Converts graphlet (snapshot) representation of temporal network and converts it to contact representation representation of network. Contact representation are more efficient for memory storing. Also includes metadata which can made it easier for plotting. A contact representation contains all non-zero edges.

    Parameters
    ----------
    G : array_like
        Temporal network.
    params : dict, optional
        Dictionary of parameters for contact representation.

        *Fs* : int, default=1
            sampling rate.

        *timeunit* : str, default=''
            Sampling rate in for units (e.g. seconds, minutes, years).

        *nettype* : str, default='auto'
            Define what type of network. Can be:
            'auto': detects automatically;
            'wd': weighted, directed;
            'bd': binary, directed;
            'wu': weighted, undirected;
            'bu': binary, undirected.

        *diagonal* : int, default = 0.
            What should the diagonal be. (note: does could be expanded to take vector of unique diagonal values in the future, but not implemented now)

        *timetype* : str, default='discrete'
            Time units can be The params file becomes the foundation of 'C'. Any other information in params, will added to C.

        *nLabs* : list
            Set nod labels.

        *t0*: int
            Time label at first index.


    Returns
    -------

    C : dict

        Contact representation of temporal network.
        Includes 'contacts', 'values' (if nettype[0]='w'),'nettype','netshape', 'Fs', 'dimord' and 'timeunit', 'timetype'.

    """

    # Create config dictionary if missing
    if params == None:
        params = {}
    # Check that temporal network is vald input.
    if G.shape[0] != G.shape[1]:
        raise ValueError(
            'Input G (node x node x time), requires Rows and Columns to be the same size.')
    if len(G.shape) == 2:
        G = np.atleast_3d(G)
    if len(G.shape) != 3:
        raise ValueError(
            'Input G must be three dimensions (node x node x time)')
    # Check number of nodes is correct, if specfied
    if 'nLabs' in params.keys():
        if params['nLabs']:
            if len(params['nLabs']) != G.shape[0]:
                raise ValueError(
                    'Specified list of node names has to be equal in length to number of nodes')
    if 't0' in params.keys():
        params['t0'] = np.atleast_1d(np.array(params['t0']))
        if len(params['t0']) != 1:
            raise ValueError(
                't0 must be sigular be either integer representing time at first temporal index)')
        params['t0'] = np.squeeze(params['t0'])
    # Check that all inputs in params are correct.

    if 'nettype' not in params.keys() or params['nettype'] == 'auto':
        params['nettype'] = gen_nettype(G, 1)
    if params['nettype'] not in {'bd', 'bu', 'wd', 'wu', 'auto'}:
        raise ValueError(
            '\'nettype\' (in params) must be a string \'wd\',\'bd\',\'wu\',\'bu\'). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if 'Fs' not in params.keys():
        params['Fs'] = 1
        print('Warning, no sampling rate set. Assuming 1.')
    if 'timeunit' not in params.keys():
        params['timeunit'] = ''
    if 'diagonal' not in params.keys():
        params['diagonal'] = 0
    if 'nLabs' not in params.keys():
        params['nLabs'] = ''
    else:
        params['nLabs'] = list(params['nLabs'])

    if 't0' not in params.keys():
        params['t0'] = 1
    nt = params['nettype']

    # Set diagonal to 0 to make contacts 0.
    G = set_diagonal(G, 0)

    # Very convoluted way to get all the indexes into a tuple, ordered by time
    if nt[1] == 'u':
        G = [np.triu(G[:, :, t], k=1) for t in range(0, G.shape[2])]
        G = np.transpose(G, [1, 2, 0])
    edg = np.where(np.abs(G) > 0)
    sortTime = np.argsort(edg[2])
    contacts = np.array([tuple([edg[0][i], edg[1][i], edg[2][i]])
                         for i in sortTime])
    # Get each of the values if weighted matrix
    if nt[0] == 'w':
        values = list(G[edg[0][sortTime], edg[1][sortTime], edg[2][sortTime]])

    # build output dictionary
    C = params
    C['contacts'] = contacts
    C['netshape'] = G.shape
    C['dimord'] = 'node,node,time'
    # Obviously this needs to change
    C['timetype'] = 'discrete'
    if nt[0] == 'w':
        C['values'] = values

    return C


def contact2graphlet(C):
    """

    Converts contact representation to graphlet (snaptshot) representation. Graphlet representation discards all meta information in the contact representation.

    Parameters
    ----------

    C : dict
        A contact representation. Must include keys: 'dimord', 'netshape', 'nettype', 'contacts' and, if weighted, 'values'.

    Returns
    -------

    G: array
        Graphlet representation of temporal network.

    Note
    ----

    Returning elements of G will be float, even if binary graph.

    """

    # Check that contact sequence is vald input.
    if 'dimord' not in C.keys():
        raise ValueError('\'dimord\' must be present in C.')
    if C['dimord'] != 'node,node,time':
        raise ValueError('\'dimord\' must be string \'node,node,time\'.')
    if 'dimord' not in C.keys():
        raise ValueError('\'dimord\' must be present in C.')
    if C['dimord'] != 'node,node,time':
        raise ValueError('\'dimord\' must be string \'node,node,time\'.')
    if 'nettype' not in C.keys():
        raise ValueError(
            'C must include parameter \'nettype\' (wd,bd,wu,bu). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if C['nettype'] not in {'bd', 'bu', 'wd', 'wu'}:
        raise ValueError(
            '\'nettype\' in (C) must be a string \'wd\',\'bd\',\'wu\',\'bu\'). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if 'netshape' not in C.keys():
        raise ValueError(
            'C must include netshape expressing size of target network (tuple)')
    if not isinstance(C['netshape'], tuple):
        raise ValueError('\'netshape\' (in C) should be a tuple')
    if len(C['netshape']) != 3:
        raise ValueError('\'netshape\' tuple should be of 3 dimensions')
    if C['nettype'][0] == 'w' and 'values' not in C.keys():
        raise ValueError('values not in C and asked for weighted network')
    if 'contacts' not in C.keys():
        raise ValueError('contacts must be expressed (list of tuples)')
    if C['timetype'] != 'discrete':
        print('Warning: timetype is not discrete. In future updates timetype in dictionary should be \'discrete\' to be converted to grpahlets')

    nt = C['nettype']

    # Preallocate
    G = np.zeros(C['netshape'])

    # Convert indexes of C to numpy friend idx list
    idx = np.array(list(map(list, C['contacts'])))
    if nt[0] == 'b':
        G[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        if nt[1] == 'u':
            G[idx[:, 1], idx[:, 0], idx[:, 2]] = 1
    elif nt[0] == 'w':
        G[idx[:, 0], idx[:, 1], idx[:, 2]] = C['values']
        if nt[1] == 'u':
            G[idx[:, 1], idx[:, 0], idx[:, 2]] = C['values']
    # If diagonal is not 0, fill it to whatever it is set to
    if C['diagonal'] != 0:
        G = set_diagonal(G, C['diagonal'])

    return G


def binarize_percent(netin, level, sign='pos', axis='time'):
    """
    Binarizes a network proprtionally. When axis='time' (only one available at the moment) then the top values for each edge time series are considered.

    Parameters
    ----------

    netin : array or dict
        network (graphlet or contact representation),
    level : float
        Percent to keep (expressed as decimal, e.g. 0.1 = top 10%)
    sign : str, default='pos'
        States the sign of the thresholding. Can be 'pos', 'neg' or 'both'. If "neg", only negative values are thresholded and vice versa.
    axis : str, default='time'
        Specify which dimension thresholding is applied against. Only 'time' option exists at present.

    Returns
    -------

    netout : array or dict (depending on input)
        Binarized network

    """
    netin, netinfo = teneto.utils.process_input(netin, ['C', 'G', 'TO'])
    if sign == 'both':
        net_sorted = np.argsort(np.abs(netin),axis=-1)
    elif sign == 'pos':
        net_sorted = np.argsort(netin,axis=-1)
    elif sign == 'neg':
        net_sorted = np.argsort(-1*netin,axis=-1)
    else:
        raise ValueError('Unknown value for parameter: sign')
    # Predefine
    netout = np.zeros(netinfo['netshape'])
    # These for loops can probabaly be removed for speed
    for i in range(netinfo['netshape'][0]):
        for j in range(netinfo['netshape'][1]):
            netout[i,j,net_sorted[i,j,-int(round(net_sorted.shape[-1])*level):]] = 1
    # Set diagonal to 0
    netout = teneto.utils.set_diagonal(netout,0)

    # If input is contact, output contact
    if netinfo['inputtype'] == 'C':
        netinfo['nettype'] = 'b' + netinfo['nettype'][1]
        netout = teneto.utils.graphlet2contact(netout,netinfo)
        netout.pop('inputtype')
        netout.pop('values')
        netout['diagonal'] = 0

    return netout


# To do: set diagonal to 0.
def binarize_rdp(netin, level, sign='pos', axis='time'):
    """
    Binarizes a network based on RDP compression.

    Parameters
    ----------

    netin : array or dict
        Network (graphlet or contact representation),
    level : float
        Delta parameter which is the tolorated error in RDP compression.
    sign : str, default='pos'
        States the sign of the thresholding. Can be 'pos', 'neg' or 'both'. If "neg", only negative values are thresholded and vice versa.

    Returns
    -------

    netout : array or dict (dependning on input)
        Binarized network
    """
    netin, netinfo = teneto.utils.process_input(netin, ['C', 'G', 'TO'])
    trajectory = teneto.trajectory.rdp(netin,level)

    contacts = []
    # Use the trajectory points as threshold
    for n in range(trajectory['index'].shape[0]):
        if sign == 'pos':
            sel = trajectory['trajectory_points'][n][trajectory['trajectory'][n][trajectory['trajectory_points'][n]]>0]
        elif sign == 'neg':
            sel = trajectory['trajectory_points'][n][trajectory['trajectory'][n][trajectory['trajectory_points'][n]]<0]
        else:
            sel = trajectory['trajectory_points']
        i_ind = np.repeat(trajectory['index'][n,0],len(sel))
        j_ind = np.repeat(trajectory['index'][n,1],len(sel))
        contacts.append(np.array([i_ind,j_ind,sel]).transpose())
    contacts = np.concatenate(contacts)

    # Create output dictionary
    netout = dict(netinfo)
    netout['contacts'] = contacts
    netout['nettype'] = 'b' + netout['nettype'][1]
    netout['dimord'] = 'node,node,time'
    netout['timetype'] = 'discrete'
    netout['diagonal'] = 0
    # If input is graphlet, output graphlet
    if netinfo['inputtype'] == 'G':
        netout = teneto.utils.contact2graphlet(netout)
    else:
        netout.pop('inputtype')

    return netout

def binarize_magnitude(netin, level, sign='pos'):
    """

    Parameters
    ----------

    netin : array or dict
        Network (graphlet or contact representation),
    level : float
        Magnitude level threshold at.
    sign : str, default='pos'
        States the sign of the thresholding. Can be 'pos', 'neg' or 'both'. If "neg", only negative values are thresholded and vice versa.
    axis : str, default='time'
        Specify which dimension thresholding is applied against. Only 'time' option exists at present.

    Returns
    -------

    netout : array or dict (depending on input)
        Binarized network
    """
    netin, netinfo = teneto.utils.process_input(netin, ['C', 'G', 'TO'])
    # Predefine
    netout = np.zeros(netinfo['netshape'])

    if sign == 'pos' or sign == 'both':
        netout[netin>level] = 1
    if sign == 'neg' or sign == 'both':
        netout[netin<level] = 1

    # Set diagonal to 0
    netout = teneto.utils.set_diagonal(netout,0)

    # If input is contact, output contact
    if netinfo['inputtype'] == 'C':
        netinfo['nettype'] = 'b' + netinfo['nettype'][1]
        netout = teneto.utils.graphlet2contact(netout,netinfo)
        netout.pop('inputtype')
        netout.pop('values')
        netout['diagonal'] = 0

    return netout

def binarize(netin, threshold_type, threshold_level, sign='pos'):
    """
    Binarizes a network, returning the network. General wrapper function for different binarization functions.

    Parameters
    ----------

    netin : array or dict
       Network (graphlet or contact representation),

    threshold_type : str
        What type of thresholds to make binarization. Options: 'rdp', 'percent', 'magnitude'.

    threshold_level : str
        Paramter dependent on threshold type.
        If 'rdp', it is the delta (i.e. error allowed in compression).
        If 'percent', it is the percentage to keep (e.g. 0.1, means keep 10% of signal).
        If 'magnitude', it is the amplitude of signal to keep.

    sign : str, default='pos'
        States the sign of the thresholding. Can be 'pos', 'neg' or 'both'. If "neg", only negative values are thresholded and vice versa.

    Returns
    -------

    netout : array or dict (depending on input)
        Binarized network

    """
    if threshold_type == 'percent':
        netout = teneto.utils.binarize_percent(netin,threshold_level,sign)
    elif threshold_type == 'magnitude':
        netout = teneto.utils.binarize_magnitude(netin,threshold_level,sign)
    elif threshold_type == 'rdp':
        netout = teneto.utils.binarize_rdp(netin,threshold_level,sign)
    else:
        raise ValueError('Unknown value to parameter: threshold_type.')
    return netout

def set_diagonal(G, val=0):
    """

    Generally diagonal is set to 0. This function helps set the diagonal across time.

    Parameters
    ----------

    G : array
        temporal network (graphlet)
    val : value to set diagonal to (default 0).

    Returns
    -------

    G : array
        Graphlet representation with new diagonal

    """

    for t in range(0, G.shape[2]):
        np.fill_diagonal(G[:, :, t], val)
    return G


def gen_nettype(G, printWarning=0):
    """

    Attempts to identify what nettype input graphlet G is. Diagonal is ignored.

    Paramters
    ---------

    G : array
        temporal network (graphlet)

    printWarning : int, default=0
        Options: 0 (default) or 1. Prints warning in console so user knows assumptions made about inputted data.

    """

    if set(np.unique(G)) == set([0, 1]):
        weights = 'b'
    else:
        weights = 'w'

    if np.allclose(G.transpose(1, 0, 2), G):
        direction = 'u'
    else:
        direction = 'd'

    nettype = weights + direction

    if printWarning == 1:
        netNames = {'w': 'weighted', 'b': 'binary',
                    'u': 'undirected', 'd': 'directed'}
        print('Assuming network is ' +
              netNames[nettype[0]] + ' and ' + netNames[nettype[1]] + '.')

    return nettype


def checkInput(netIn, raiseIfU=1, conMat=0):
    """

    This function checks that netIn input is either graphlet (G) or contact (C).

    Parameters
    ----------

    netIn : array or dict
        temporal network, (graphlet or contact).
    raiseIfU : int, default=1.
        Options 1 or 0. Error is raised if not found to be G or C
    conMat : int, default=0.
        Options 1 or 0. If 1, input is allowed to be a 2 dimensional connectivity matrix. Allows output to be 'M'

    Returns
    -------

    inputtype : str
        String indicating input type. 'G','C', 'M' or 'U' (unknown). M is special case only allowed when conMat=1 for a 2D connectivity matrix.

    """

    inputIs = 'U'
    if isinstance(netIn, np.ndarray):
        netShape = netIn.shape
        if len(netShape) == 3 and netShape[0] == netShape[1]:
            inputIs = 'G'
        if netShape[0] == netShape[1] and conMat == 1:
            inputIs = 'M'

    elif isinstance(netIn, dict):
        if 'nettype' in netIn and 'contacts' in netIn and 'dimord' in netIn and 'timetype' in netIn:
            if netIn['nettype'] in {'bd', 'bu', 'wd', 'wu'} and netIn['timetype'] == 'discrete' and netIn['dimord'] == 'node,node,time':
                inputIs = 'C'

    elif isinstance(netIn, object):
        if isinstance(netIn.contact, dict):
            if 'nettype' in netIn.contact and 'contacts' in netIn.contact and 'dimord' in netIn.contact and 'timetype' in netIn.contact:
                if netIn.contact['nettype'] in {'bd', 'bu', 'wd', 'wu'} and netIn.contact['timetype'] == 'discrete' and netIn.contact['dimord'] == 'node,node,time':
                    inputIs = 'TO'

    if raiseIfU == 1 and inputIs == 'U':
        raise ValueError(
            'Input cannot be identified as graphlet or contact representation')

    return inputIs


def getDistanceFunction(requested_metric):
    """

    This function returns a specified distance function.


    Paramters
    ---------

    requested_metric: str
        Distance function. Can be any function in: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html.

    Returns
    -------

    requested_metric : distance function

    """

    distance_options = {
        'braycurtis': distance.braycurtis,
        'canberra': distance.canberra,
        'chebyshev': distance.chebyshev,
        'cityblock': distance.cityblock,
        'correlation': distance.correlation,
        'cosine': distance.cosine,
        'euclidean': distance.euclidean,
        'sqeuclidean': distance.sqeuclidean,
        'dice': distance.dice,
        'hamming': distance.hamming,
        'jaccard': distance.jaccard,
        'kulsinski': distance.kulsinski,
        'matching': distance.matching,
        'rogerstanimoto': distance.rogerstanimoto,
        'russellrao': distance.russellrao,
        'sokalmichener': distance.sokalmichener,
        'sokalsneath': distance.sokalsneath,
        'yule': distance.yule,
    }

    if requested_metric in distance_options:
        return distance_options[requested_metric]
    else:
        raise ValueError('Distance function cannot be found.')


def process_input(netIn, allowedformats, outputformat='G'):
    """
    Takes input network and checks what the input is.

    Parameters
    ----------

    netIn : array, dict, or class
        Network (graphlet, contact or object)
    allowedformats : str
        Which format of network objects that are allowed. Options: 'C', 'TO', 'G'.
    outputformat: str, default=G
        Target output format. Options: 'C' or 'G'.

    Returns
    -------

    C : dict

    OR

    G : array
        Graphlet representation.
    netInfo : dict
        Metainformation about network.

    """
    inputtype = teneto.utils.checkInput(netIn)
    # Convert TO to C representation
    if inputtype == 'TO' and 'TO' in allowedformats:
        G = netIn.get_graphlet_representation()
        netInfo = dict(netIn.contact)
        netInfo.pop('contacts')
    # Convert C representation to G
    elif inputtype == 'C' and 'C' in allowedformats and outputformat != 'C':
        G = teneto.utils.contact2graphlet(netIn)
        netInfo = dict(netIn)
        netInfo.pop('contacts')
        nettype = netIn['nettype']
    # Get network type if not set yet
    elif inputtype == 'G' and 'G' in allowedformats:
        netInfo = {}
        netInfo['netshape'] = netIn.shape
        netInfo['nettype'] = teneto.utils.gen_nettype(netIn)
        G = netIn
    elif inputtype == 'C' and outputformat == 'C':
        pass
    else:
        raise ValueError('Input invalid.')
    netInfo['inputtype'] = inputtype
    if inputtype != 'C' and outputformat == 'C':
        C = teneto.utils.graphlet2contact(netIn, netInfo)
    if outputformat == 'G':
        return G, netInfo
    elif outputformat == 'C':
        return C


def clean_community_indexes(communityID):
    """
    Takes input of community assignments. Returns reindexed community assignment by using smallest numbers possible.

    Parameters
    ----------

    communityID : array-like
        list or array of integers. Output from community detection algorithems.

    Returns
    -------

    new_communityID : array
        cleaned list going from 0 to len(np.unique(communityID))-1

    Note
    -----

    Behaviour of funciton entails that the lowest community integer in communityID will recieve the lowest integer in new_communityID.

    """
    communityID = np.array(communityID)
    cid_shape = communityID.shape
    if len(cid_shape) > 1: 
        communityID = communityID.flatten()  
    new_communityID = np.zeros(len(communityID))
    for i, n in enumerate(np.unique(communityID)):
        new_communityID[communityID == n] = i
    if len(cid_shape) > 1: 
        new_communityID = new_communityID.reshape(cid_shape)  
    return new_communityID


def multiple_contacts_get_values(C):
    """
    Given an contact representation with repeated contacts, this function removes duplicates and creates a value

    Parameters
    ----------

    C : dict

        contact representation with multiple repeated contacts.

    Returns
    -------

    :C_out: dict

        Contact representation with duplicate contacts removed and the number of duplicates is now in the 'values' field.

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
    C_out['contacts'] = new_contacts
    C_out['values'] = new_values
    return C_out



def check_distance_funciton_input(distance_func_name,netinfo):
    """
    Funciton checks distance_func_name, if it is specified as 'default'. Then given the type of the network selects a default distance function.

    Parameters
    ----------

    distance_func_name : str
        distance function name.

    netinfo : dict
        the output of utils.process_input

    Returns
    -------

    distance_func_name : str
        distance function name.
    """

    if distance_func_name == 'default' and netinfo['nettype'][0] == 'b':
        print('Default distance funciton specified. As network is binary, using Hamming')
        distance_func_name = 'hamming'
    elif distance_func_name == 'default' and netinfo['nettype'][0] == 'w':
        distance_func_name = 'euclidean'
        print(
            'Default distance funciton specified. '
            'As network is weighted, using Euclidean')

    return distance_func_name




def load_parcellation_coords(parcellation_name):
    """
    Loads coordinates of included parcellations.

    Parameters
    ----------

    parcellation_name : str
        options: 'gordon2014_333', 'power2012_264', 'shen2013_278'.

    Returns
    -------
    parc : array
        parcellation cordinates

    """

    path = teneto.__path__[0] + '/data/parcellation/' + parcellation_name + '.csv'
    parc = np.loadtxt(path,skiprows=1,delimiter=',',usecols=[1,2,3])

    return parc


def make_parcellation(data_path, parcellation, parc_type=None, parc_params=None):

    """
    Performs a parcellation which reduces voxel space to regions of interest (brain data).

    Parameters
    ----------

    data_path : str
        Path to .nii image.
    parcellation : str
        Specify which parcellation that you would like to use. For MNI: 'gordon2014_333', 'power2012_264', For TAL: 'shen2013_278'.
        It is possible to add the OH subcotical atlas on top of a cortical atlas (e.g. gordon) by adding:
            '+sub-maxprob-thr0-1mm', '+sub-maxprob-thr0-2mm', 'sub-maxprob-thr25-1mm', 'sub-maxprob-thr25-2mm',
            '+sub-maxprob-thr50-1mm', '+sub-maxprob-thr50-2mm'.
            e.g.: gordon2014_333+submaxprob-thr0-2mm'
    parc_type : str
        Can be 'sphere' or 'region'. If nothing is specified, the default for that parcellation will be used.
    parc_params : dict
        **kwargs for nilearn functions

    Returns
    -------

    data : array
        Data after the parcellation.

    NOTE
    ----
    These functions make use of nilearn. Please cite nilearn if used in a publicaiton.
    """

    if isinstance(parcellation,str):

        if '+' in parcellation:
            parcin = parcellation.split('+')
            parcellation = parcin[0]
            subcortical = parcin[1]
        else:
            subcortical = None

        if not parc_type or not parc_params:
            path = teneto.__path__[0] + '/data/parcellation_defaults/defaults.json'
            with open(path) as data_file:
                defaults = json.load(data_file)
        if not parc_type:
            parc_type = defaults[parcellation]['type']
            print('Using default parcellation type')
        if not parc_params:
            parc_params = defaults[parcellation]['params']
            print('Using default parameters')

    if parc_type == 'sphere':
        parcellation = teneto.utils.load_parcellation_coords(parcellation)
        seed = NiftiSpheresMasker(np.array(parcellation),**parc_params)
        data = seed.fit_transform(data_path)
    elif parc_type == 'region':
        path = teneto.__path__[0] + '/data/parcellation/' + parcellation + '.nii'
        region = NiftiLabelsMasker(path,**parc_params)
        data = region.fit_transform(data_path)
    else:
        raise ValueError('Unknown parc_type specified')

    if subcortical:
        subatlas = fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm')['maps']
        region = NiftiLabelsMasker(subatlas,**parc_params)
        data_sub = region.fit_transform(data_path)
        data = np.hstack([data, data_sub])

    return data



def create_traj_ranges(start, stop, N):

    """
    Fills in the trajectory range.

    # Adapted from https://stackoverflow.com/a/40624614
    """
    steps = (1.0/(N-1)) * (stop - start)
    if np.isscalar(steps):
        return steps*np.arange(N) + start
    else:
        return steps[:,None]*np.arange(N) + start[:,None]



def get_dimord(measure,calc=None,subnet=None):

    """
    Get the dimension order of a network measure.

    Parameters
    ----------

    measure : str
        Name of funciton in teneto.networkmeasures.
    calc : str, default=None
        Calc parameter for the function
    subnet : bool, default=None
        If not null, then subnet property is assumed to be believed.

    Returns
    -------

    dimord : str
        Dimension order. So "node,node,time" would define the dimensions of the network measure.

    """

    if not calc:
        calc = ''
    else:
        calc = '_' + calc
    if not subnet:
        subnet = ''
    else:
        subnet = '_subnet'
    if '_subnet' in calc and '_subnet' in subnet:
        subnet = ''
    if calc == 'subnet_avg' or calc == 'subnet_pairs':
        subnet = ''

    dimord_dict = {
        'temporal_closeness_centrality': 'node',
        'temporal_degree_centrality': 'node',
        'temporal_degree_centralit_avg': 'node',
        'temporal_degree_centrality_time': 'node,time',
        'temporal_degree_centrality_time_subnet': 'subnet,subnet,time',
        'temporal_degree_centrality_subnet': 'subnet,subnet',
        'temporal_degree_centrality_avg_subnet': 'subnet,subnet',
        'temporal_efficiency': 'global',
        'temporal_efficiency_global': 'global',
        'temporal_efficiency_node': 'node',
        'temporal_efficiency_to': 'node',
        'sid_global': 'global,time',
        'sid_global_subnet': 'global,time',
        'sid_subnet_pairs': 'subnet,subnet,time',
        'sid_subnet_avg': 'subnet,time',
        'sid': 'subnet,subnet,time',
        'reachability_latency_global': 'global',
        'reachability_latency': 'global',
        'reachability_latency_node': 'node',
        'fluctuability': 'node',
        'fluctuability_global': 'global',
        'bursty_coeff': 'edge,edge',
        'bursty_coeff_edge': 'edge,edge',
        'bursty_coeff_node': 'node',
        'bursty_coeff_subnet': 'subnet,subnet',
        'bursty_coeff_meanEdgePerNode': 'node',
        'volatility_global': 'time',
    }
    if measure + calc + subnet in dimord_dict:
        return dimord_dict[measure + calc + subnet]
    else:
        print('WARNINGL: get_dimord() returned unknown dimension labels')
        return 'unknown'
