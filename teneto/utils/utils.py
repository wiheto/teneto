"""General utility functions"""

import numpy as np
import collections
import itertools
import scipy.spatial.distance as distance
from ..trajectory import rdp
from ..classes import TemporalNetwork
import pandas as pd
import operator


def graphlet2contact(tnet, params=None):
    """

    Converts array representation to contact representation.

    Contact representation are more efficient for memory storing.
    Also includes metadata which can made it easier for plotting.
    A contact representation contains all non-zero edges.

    Parameters
    ----------
    tnet : array_like
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
            What should the diagonal be.

        *timetype* : str, default='discrete'
            Time units can be The params file becomes the foundation of 'C'.
            Any other information in params, will added to C.

        *nodelabels* : list
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
    if params is None:
        params = {}
    # Check that temporal network is vald input.
    if tnet.shape[0] != tnet.shape[1]:
        raise ValueError(
            'Input tnet (node x node x time), requires Rows and Columns to be the same size.')
    if len(tnet.shape) == 2:
        tnet = np.atleast_3d(tnet)
    if len(tnet.shape) != 3:
        raise ValueError(
            'Input tnet must be three dimensions (node x node x time)')
    # Check number of nodes is correct, if specfied
    if 'nodelabels' in params.keys():
        if params['nodelabels']:
            if len(params['nodelabels']) != tnet.shape[0]:
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
        params['nettype'] = gen_nettype(tnet, 1)
    if params['nettype'] not in {'bd', 'bu', 'wd', 'wu', 'auto'}:
        raise ValueError(
            '\'nettype\' (in params) must be a string \'wd\',\'bd\',\'wu\',\'bu\').')
    if 'Fs' not in params.keys():
        params['Fs'] = 1
        #print('Warning, no sampling rate set. Assuming 1.')
    if 'timeunit' not in params.keys():
        params['timeunit'] = ''
    if 'diagonal' not in params.keys():
        params['diagonal'] = 0
    if 'nodelabels' not in params.keys():
        params['nodelabels'] = ''
    else:
        params['nodelabels'] = list(params['nodelabels'])

    if 't0' not in params.keys():
        params['t0'] = 1
    nt = params['nettype']

    # Set diagonal to 0 to make contacts 0.
    tnet = set_diagonal(tnet, 0)

    # Very convoluted way to get all the indexes into a tuple, ordered by time
    if nt[1] == 'u':
        tnet = [np.triu(tnet[:, :, t], k=1) for t in range(0, tnet.shape[2])]
        tnet = np.transpose(tnet, [1, 2, 0])
    edg = np.where(np.abs(tnet) > 0)
    sortTime = np.argsort(edg[2])
    contacts = np.array([tuple([edg[0][i], edg[1][i], edg[2][i]])
                         for i in sortTime])
    # Get each of the values if weighted matrix
    if nt[0] == 'w':
        values = list(tnet[edg[0][sortTime], edg[1]
                           [sortTime], edg[2][sortTime]])

    # build output dictionary
    C = params
    C['contacts'] = contacts
    C['netshape'] = tnet.shape
    C['dimord'] = 'node,node,time'
    # Obviously this needs to change
    C['timetype'] = 'discrete'
    if nt[0] == 'w':
        C['values'] = values

    return C


def contact2graphlet(C):
    """

    Converts contact representation to array representation.

    Graphlet representation discards all meta information in contacts.

    Parameters
    ----------

    C : dict
        A contact representation. Must include keys: 'dimord', 'netshape', 'nettype', 'contacts' and, if weighted, 'values'.

    Returns
    -------

    tnet : array
        Graphlet representation of temporal network.

    Note
    ----

    Returning elements of tnet will be float, even if binary graph.

    """
    # Check that contact sequence is vald input.
    if 'dimord' not in C.keys():
        raise ValueError('\'dimord\' must be present in C.')
    if C['dimord'] != 'node,node,time':
        raise ValueError('\'dimord\' must be string \'node,node,time\'.')
    if 'nettype' not in C.keys():
        raise ValueError(
            'C must include parameter \'nettype\' (wd,bd,wu,bu). w: weighted network. b: binary network. u: undirected network. d: directed network')
    if C['nettype'] not in {'bd', 'bu', 'wd', 'wu'}:
        raise ValueError(
            '\'nettype\' in (C) must be a string \'wd\',\'bd\',\'wu\',\'bu\').')
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
    tnet = np.zeros(C['netshape'])

    # Convert indexes of C to numpy friend idx list
    idx = np.array(list(map(list, C['contacts'])))
    if nt[0] == 'b':
        tnet[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        if nt[1] == 'u':
            tnet[idx[:, 1], idx[:, 0], idx[:, 2]] = 1
    elif nt[0] == 'w':
        tnet[idx[:, 0], idx[:, 1], idx[:, 2]] = C['values']
        if nt[1] == 'u':
            tnet[idx[:, 1], idx[:, 0], idx[:, 2]] = C['values']
    # If diagonal is not 0, fill it to whatever it is set to
    if C['diagonal'] != 0:
        tnet = set_diagonal(tnet, C['diagonal'])

    return tnet


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
        Specify which dimension thresholding is applied against.
        Can be 'time' (takes top % for each edge time-series) or 'graphlet'
        (takes top % for each graphlet)

    Returns
    -------

    netout : array or dict (depending on input)
        Binarized network

    """
    netin, netinfo = process_input(netin, ['C', 'G', 'TN'])
    # Set diagonal to 0
    netin = set_diagonal(netin, 0)
    if axis == 'graphlet' and netinfo['nettype'][-1] == 'u':
        triu = np.triu_indices(netinfo['netshape'][0], k=1)
        netin = netin[triu[0], triu[1], :]
        netin = netin.transpose()
    if sign == 'both':
        net_sorted = np.argsort(np.abs(netin), axis=-1)
    elif sign == 'pos':
        net_sorted = np.argsort(netin, axis=-1)
    elif sign == 'neg':
        net_sorted = np.argsort(-1*netin, axis=-1)
    else:
        raise ValueError('Unknown value for parameter: sign')
    # Predefine
    netout = np.zeros(netinfo['netshape'])
    if axis == 'time':
        # These for loops can probabaly be removed for speed
        for i in range(netinfo['netshape'][0]):
            for j in range(netinfo['netshape'][1]):
                netout[i, j, net_sorted[i, j, -
                                        int(round(net_sorted.shape[-1])*level):]] = 1
    elif axis == 'graphlet':
        netout_tmp = np.zeros(netin.shape)
        for i in range(netout_tmp.shape[0]):
            netout_tmp[i, net_sorted[i, -
                                     int(round(net_sorted.shape[-1])*level):]] = 1
        netout_tmp = netout_tmp.transpose()
        netout[triu[0], triu[1], :] = netout_tmp
        netout[triu[1], triu[0], :] = netout_tmp

    netout = set_diagonal(netout, 0)

    # If input is contact, output contact
    if netinfo['inputtype'] == 'C':
        netinfo['nettype'] = 'b' + netinfo['nettype'][1]
        netout = graphlet2contact(netout, netinfo)
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
    netin, netinfo = process_input(netin, ['C', 'G', 'TN'])
    trajectory = rdp(netin, level)

    contacts = []
    # Use the trajectory points as threshold
    for n in range(trajectory['index'].shape[0]):
        if sign == 'pos':
            sel = trajectory['trajectory_points'][n][trajectory['trajectory']
                                                     [n][trajectory['trajectory_points'][n]] > 0]
        elif sign == 'neg':
            sel = trajectory['trajectory_points'][n][trajectory['trajectory']
                                                     [n][trajectory['trajectory_points'][n]] < 0]
        else:
            sel = trajectory['trajectory_points']
        i_ind = np.repeat(trajectory['index'][n, 0], len(sel))
        j_ind = np.repeat(trajectory['index'][n, 1], len(sel))
        contacts.append(np.array([i_ind, j_ind, sel]).transpose())
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
        netout = contact2graphlet(netout)
    else:
        netout.pop('inputtype')

    return netout


def binarize_magnitude(netin, level, sign='pos'):
    """
    Make binary network based on magnitude thresholding.

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
    netin, netinfo = process_input(netin, ['C', 'G', 'TN'])
    # Predefine
    netout = np.zeros(netinfo['netshape'])

    if sign == 'pos' or sign == 'both':
        netout[netin > level] = 1
    if sign == 'neg' or sign == 'both':
        netout[netin < level] = 1

    # Set diagonal to 0
    netout = set_diagonal(netout, 0)

    # If input is contact, output contact
    if netinfo['inputtype'] == 'C':
        netinfo['nettype'] = 'b' + netinfo['nettype'][1]
        netout = graphlet2contact(netout, netinfo)
        netout.pop('inputtype')
        netout.pop('values')
        netout['diagonal'] = 0

    return netout


def binarize(netin, threshold_type, threshold_level, outputformat='auto', sign='pos', axis='time'):
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

    outputformat : str
        specify what format you want the output in: G, C, TN, or DF. If 'auto', input form is returned.

    sign : str, default='pos'
        States the sign of the thresholding. Can be 'pos', 'neg' or 'both'. If "neg", only negative values are thresholded and vice versa.

    axis : str
        Threshold over specfied axis. Valid for percent and rdp. Can be time or graphlet.

    Returns
    -------

    netout : array or dict (depending on input)
        Binarized network

    """
    if outputformat == 'auto':
        outputformat = check_input(netin)
    if threshold_type == 'percent':
        netout = binarize_percent(netin, threshold_level, sign, axis)
    elif threshold_type == 'magnitude':
        netout = binarize_magnitude(netin, threshold_level, sign)
    elif threshold_type == 'rdp':
        netout = binarize_rdp(netin, threshold_level, sign, axis)
    else:
        raise ValueError('Unknown value to parameter: threshold_type.')
    netout = process_input(netout, ['G'], outputformat=outputformat)
    if outputformat == 'G':
        netout = netout[0]
    return netout


def set_diagonal(tnet, val=0):
    """

    Generally diagonal is set to 0. This function helps set the diagonal across time.

    Parameters
    ----------

    tnet : array
        temporal network (graphlet)
    val : value to set diagonal to (default 0).

    Returns
    -------

    tnet : array
        Graphlet representation with new diagonal

    """
    for t in range(0, tnet.shape[2]):
        np.fill_diagonal(tnet[:, :, t], val)
    return tnet


def gen_nettype(tnet, printWarning=0, weightonly=False):
    r"""

    Attempts to identify what nettype input graphlet tnet is. Diagonal is ignored.

    Paramters
    ---------

    tnet : array
        temporal network (graphlet)

    Returns
    -------
    nettype : str
        \'wu\', \'bu\', \'wd\', or \'bd\'
    """
    if np.array_equal(tnet, tnet.astype(bool)):
        nettype = 'b'
    else:
        nettype = 'w'

    if not weightonly:
        if np.allclose(tnet.transpose(1, 0, 2), tnet):
            direction = 'u'
        else:
            direction = 'd'

        nettype = nettype + direction

    return nettype


def check_input(netin, raiseIfU=1, conMat=0):
    """

    This function checks that netin input is either graphlet (tnet) or contact (C).

    Parameters
    ----------

    netin : array or dict
        temporal network, (graphlet or contact).
    raiseIfU : int, default=1.
        Options 1 or 0. Error is raised if not found to be tnet or C
    conMat : int, default=0.
        Options 1 or 0. If 1, input is allowed to be a 2 dimensional connectivity matrix. Allows output to be 'M'

    Returns
    -------

    inputtype : str
        String indicating input type. 'G','C', 'M' or 'U' (unknown). M is special case only allowed when conMat=1 for a 2D connectivity matrix.

    """
    inputis = 'U'
    if isinstance(netin, np.ndarray):
        netShape = netin.shape
        if len(netShape) == 3 and netShape[0] == netShape[1]:
            inputis = 'G'
        elif netShape[0] == netShape[1] and conMat == 1:
            inputis = 'M'

    elif isinstance(netin, dict):
        if 'nettype' in netin and 'contacts' in netin and 'dimord' in netin and 'timetype' in netin:
            if netin['nettype'] in {'bd', 'bu', 'wd', 'wu'} and netin['timetype'] == 'discrete' and netin['dimord'] == 'node,node,time':
                inputis = 'C'

    elif isinstance(netin, object):
        if hasattr(netin, 'network'):
            inputis = 'TN'
        elif isinstance(netin, pd.DataFrame):
            inputis = 'DF'

    if raiseIfU == 1 and inputis == 'U':
        raise ValueError(
            'Input cannot be identified as graphlet or contact representation')

    return inputis


def get_distance_function(requested_metric):
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


def process_input(netIn, allowedformats, outputformat='G', forcesparse=False):
    """
    Takes input network and checks what the input is.

    Parameters
    ----------

    netIn : array, dict, or TemporalNetwork
        Network (graphlet, contact or object)
    allowedformats : list or str
        Which format of network objects that are allowed. Options: 'C', 'TN', 'G'.
    outputformat: str, default=G
        Target output format. Options: 'C' or 'G'.


    Returns
    -------

    C : dict

    OR

    tnet : array
        Graphlet representation.
    netinfo : dict
        Metainformation about network.

    OR

    tnet : object
        object of TemporalNetwork class

    """
    netinfo = {}
    if outputformat == 'DF':
        outputformat = 'TN'
        return_df = True
        forcesparse = True
    else:
        return_df = False
    inputtype = check_input(netIn)
    if inputtype == 'DF':
        netIn = TemporalNetwork(from_df=netIn)
        inputtype = 'TN'
    # Convert TN to tnet representation
    if inputtype == 'TN' and 'TN' in allowedformats and outputformat != 'TN':
        if netIn.sparse:
            tnet = netIn.df_to_array()
        else:
            tnet = netIn.network
        netinfo = {'nettype': netIn.nettype, 'netshape': [
            netIn.netshape[0], netIn.netshape[0], netIn.netshape[1]]}
    elif inputtype == 'TN' and 'TN' in allowedformats and outputformat == 'TN':
        if not netIn.sparse and forcesparse:
            TN = TemporalNetwork(from_array=netIn.network, forcesparse=True)
        else:
            TN = netIn
    elif inputtype == 'C' and 'C' in allowedformats and outputformat == 'G':
        tnet = contact2graphlet(netIn)
        netinfo = dict(netIn)
        netinfo.pop('contacts')
    elif inputtype == 'C' and 'C' in allowedformats and outputformat == 'TN':
        TN = TemporalNetwork(from_dict=netIn)
    elif inputtype == 'G' and 'G' in allowedformats and outputformat == 'TN':
        TN = TemporalNetwork(from_array=netIn, forcesparse=forcesparse)
    # Get network type if not set yet
    elif inputtype == 'G' and 'G' in allowedformats:
        netinfo = {}
        netinfo['netshape'] = netIn.shape
        netinfo['nettype'] = gen_nettype(netIn)
        tnet = netIn
    elif inputtype == 'C' and outputformat == 'C':
        pass
    else:
        raise ValueError('Input invalid.')
    if outputformat == 'TN' and isinstance(TN.network, pd.DataFrame):
        TN.network['i'] = TN.network['i'].astype(int)
        TN.network['j'] = TN.network['j'].astype(int)
        TN.network['t'] = TN.network['t'].astype(int)
    if outputformat == 'C' or outputformat == 'G':
        netinfo['inputtype'] = inputtype
    if inputtype != 'C' and outputformat == 'C':
        return graphlet2contact(tnet, netinfo)
    if outputformat == 'G':
        return tnet, netinfo
    elif outputformat == 'C':
        return netIn
    elif outputformat == 'TN':
        if return_df:
            return TN.network
        else:
            return TN


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


def df_to_array(df, netshape, nettype):
    """
    Returns a numpy array (snapshot representation) from thedataframe contact list

    Parameters:
        df : pandas df
            pandas df with columns, i,j,t.
        netshape : tuple
            network shape, format: (node, time)
        nettype : str
            'wu', 'wd', 'bu', 'bd'

    Returns:
    --------
        tnet : array
            (node,node,time) array for the network
    """
    if len(df) > 0:
        idx = np.array(list(map(list, df.values)))
        tnet = np.zeros([netshape[0], netshape[0], netshape[1]])
        if idx.shape[1] == 3:
            if nettype[-1] == 'u':
                idx = np.vstack([idx, idx[:, [1, 0, 2]]])
            idx = idx.astype(int)
            tnet[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        elif idx.shape[1] == 4:
            if nettype[-1] == 'u':
                idx = np.vstack([idx, idx[:, [1, 0, 2, 3]]])
            weights = idx[:, 3]
            idx = np.array(idx[:, :3], dtype=int)
            tnet[idx[:, 0], idx[:, 1], idx[:, 2]] = weights
    else:
        tnet = np.zeros([netshape[0], netshape[0], netshape[1]])
    return tnet


def check_distance_funciton_input(distance_func_name, netinfo):
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


def create_traj_ranges(start, stop, N):
    """
    Fills in the trajectory range.

    # Adapted from https://stackoverflow.com/a/40624614
    """
    steps = (1.0/(N-1)) * (stop - start)
    if np.isscalar(steps):
        return steps*np.arange(N) + start
    else:
        return steps[:, None]*np.arange(N) + start[:, None]


def get_dimord(measure, calc=None, community=None):
    """
    Get the dimension order of a network measure.

    Parameters
    ----------

    measure : str
        Name of funciton in teneto.networkmeasures.
    calc : str, default=None
        Calc parameter for the function
    community : bool, default=None
        If not null, then community property is assumed to be believed.

    Returns
    -------

    dimord : str
        Dimension order. So "node,node,time" would define the dimensions of the network measure.

    """
    if not calc:
        calc = ''
    else:
        calc = '_' + calc
    if not community:
        community = ''
    else:
        community = 'community'
    if 'community' in calc and 'community' in community:
        community = ''
    if calc == 'community_avg' or calc == 'community_pairs':
        community = ''

    dimord_dict = {
        'temporal_closeness_centrality': 'node',
        'temporal_degree_centrality': 'node',
        'temporal_degree_centralit_avg': 'node',
        'temporal_degree_centrality_time': 'node,time',
        'temporal_efficiency': 'global',
        'temporal_efficiency_global': 'global',
        'temporal_efficiency_node': 'node',
        'temporal_efficiency_to': 'node',
        'sid_global': 'global,time',
        'community_pairs': 'community,community,time',
        'community_avg': 'community,time',
        'sid': 'community,community,time',
        'reachability_latency_global': 'global',
        'reachability_latency': 'global',
        'reachability_latency_node': 'node',
        'fluctuability': 'node',
        'fluctuability_global': 'global',
        'bursty_coeff': 'edge,edge',
        'bursty_coeff_edge': 'edge,edge',
        'bursty_coeff_node': 'node',
        'bursty_coeff_meanEdgePerNode': 'node',
        'volatility_global': 'time',
    }
    if measure + calc + community in dimord_dict:
        return dimord_dict[measure + calc + community]
    else:
        print('WARNINGL: get_dimord() returned unknown dimension labels')
        return 'unknown'


def get_network_when(tnet, i=None, j=None, t=None, ij=None, logic='and', copy=False, asarray=False, netshape=None, nettype=None):
    r"""
    Returns subset of dataframe that matches index

    Parameters
    ----------
    tnet : df, array or TemporalNetwork
        TemporalNetwork object or pandas dataframe edgelist
    i : list or int
        get nodes in column i (source nodes in directed networks)
    j : list or int
        get nodes in column j (target nodes in directed networks)
    t : list or int
        get edges at this time-points.
    ij : list or int
        get nodes for column i or j (logic and can still persist for t). Cannot be specified along with i or j
    logic : str
        options: \'and\' or \'or\'. If \'and\', functions returns rows that corrspond that match all i,j,t arguments. If \'or\', only has to match one of them
    copy : bool
        default False. If True, returns a copy of the dataframe. Note relevant if hd5 data.
    asarray : bool
        default False. If True, returns the list of edges as a numpy array.

    Returns
    -------
    df : pandas dataframe
        Unless asarray are set to true.
    """
    if isinstance(tnet, pd.DataFrame):
        network = tnet
        hdf5 = False
        sparse = True
    elif isinstance(tnet, np.ndarray):
        network = tnet
        sparse = False
    # Can add hdfstore
    elif isinstance(tnet, object):
        network = tnet.network
        hdf5 = tnet.hdf5
        sparse = tnet.sparse
        nettype = tnet.nettype
        netshape = tnet.netshape
    if ij is not None and (i is not None or j is not None):
        raise ValueError('ij cannoed be specifed along with i or j')
    # Make non list inputs a list
    if i is not None and not isinstance(i, list):
        i = [i]
    if j is not None and not isinstance(j, list):
        j = [j]
    if t is not None and not isinstance(t, list):
        t = [t]
    if ij is not None and not isinstance(ij, list):
        ij = [ij]
    if hdf5:
        l = {'or': ' | ', 'and': ' & '}
        if i is not None and j is not None and t is not None:
            isinstr = 'i in ' + str(i) + l[logic] + 'j in ' + \
                str(j) + l[logic] + 't in ' + str(t)
        elif ij is not None and t is not None:
            isinstr = '(i in ' + str(ij) + ' | ' + 'j in ' + \
                str(ij) + ') & ' + 't in ' + str(t)
        elif i is not None and j is not None:
            isinstr = 'i in ' + str(i) + l[logic] + 'j in ' + str(j)
        elif i is not None and t is not None:
            isinstr = 'i in ' + str(i) + l[logic] + 't in ' + str(t)
        elif j is not None and t is not None:
            isinstr = 'j in ' + str(j) + l[logic] + 't in ' + str(t)
        elif i is not None:
            isinstr = 'i in ' + str(i)
        elif j is not None:
            isinstr = 'j in ' + str(j)
        elif t is not None:
            isinstr = 't in ' + str(t)
        elif ij is not None:
            isinstr = 'i in ' + str(ij) + l['or'] + 'j in ' + str(ij)
        df = pd.read_hdf(network, where=isinstr)
    elif not sparse:
        if logic == 'or':
            raise ValueError(
                'OR logic not implemented with array/dense format yet!')
        else:
            if t is None:
                t = np.arange(network.shape[-1])
            if i is None:
                i = np.arange(network.shape[0])
            if j is None:
                j = np.arange(network.shape[0])
            if ij is not None:
                i = ij
                j = np.arange(network.shape[0])
        ind = list(zip(*itertools.product(i, j, t)))
        ind = np.array(ind)
        if ij is None:
            ind2 = np.array(list(zip(*itertools.product(j, i, t))))
            ind = np.hstack([ind, ind2])

        edges = network[ind[0], ind[1], ind[2]]
        ind = ind[:, edges != 0]
        edges = edges[edges != 0]
        df = pd.DataFrame(
            data={'i': ind[0], 'j': ind[1], 't': ind[2], 'weight': edges})
        df['i'] = df['i'].astype(int)
        df['j'] = df['j'].astype(int)
        if nettype[1] == 'u':
            df = df_drop_ij_duplicates(df)

    else:
        l = {'or': operator.__or__, 'and': operator.__and__}
        if i is not None and j is not None and t is not None:
            df = network[l[logic]((network['i'].isin(i)), l[logic]((
                network['j'].isin(j)), (network['t'].isin(t))))]
        elif ij is not None and t is not None:
            df = network[((network['i'].isin(ij)) | l[logic]((
                network['j'].isin(ij)), (network['t'].isin(t))))]
        elif i is not None and j is not None:
            df = network[l[logic]((network['i'].isin(i)),
                                  (network['j'].isin(j)))]
        elif i is not None and t is not None:
            df = network[l[logic]((network['i'].isin(i)),
                                  (network['t'].isin(t)))]
        elif j is not None and t is not None:
            df = network[l[logic]((network['j'].isin(j)),
                                  (network['t'].isin(t)))]
        elif i is not None:
            df = network[network['i'].isin(i)]
        elif j is not None:
            df = network[network['j'].isin(j)]
        elif t is not None:
            df = network[network['t'].isin(t)]
        elif ij is not None:
            df = network[(network['i'].isin(ij)) | (network['j'].isin(ij))]
        if copy:
            df = df.copy()
    if asarray:
        df = df_to_array(df, netshape, nettype)
    return df


def create_supraadjacency_matrix(tnet, intersliceweight=1):
    """
    Returns a supraadjacency matrix from a temporal network structure

    Parameters
    --------
    tnet : TemporalNetwork
        Temporal network (any network type)
    intersliceweight : int
        Weight that links the same node from adjacent time-points

    Returns
    --------
    supranet : dataframe
        Supraadjacency matrix
    """
    tnet = process_input(tnet, ['G', 'C', 'TN'], 'TN', forcesparse=True)
    newnetwork = tnet.network.copy()
    newnetwork['i'] = (tnet.network['i']) + \
        ((tnet.netshape[0]) * (tnet.network['t']))
    newnetwork['j'] = (tnet.network['j']) + \
        ((tnet.netshape[0]) * (tnet.network['t']))
    if 'weight' not in newnetwork.columns:
        newnetwork['weight'] = 1
    newnetwork.drop('t', axis=1, inplace=True)
    timepointconns = pd.DataFrame()
    timepointconns['i'] = np.arange(0, (tnet.N*tnet.T)-tnet.N)
    timepointconns['j'] = np.arange(tnet.N, (tnet.N*tnet.T))
    timepointconns['weight'] = intersliceweight
    supranet = pd.concat([newnetwork, timepointconns]).reset_index(drop=True)
    return supranet


def check_TemporalNetwork_input(datain, datatype):
    """
    """
    if datatype == 'edgelist':
        if not isinstance(datain, list):
            raise ValueError('edgelist should be list')
        if all([len(e) == 3 for e in datain]) or all([len(e) == 4 for e in datain]):
            pass
        else:
            raise ValueError(
                'Each member in edgelist should all be a list of length 3 (i,j,t) or 4 (i,j,t,w)')
    elif datatype == 'array':
        if not isinstance(datain, np.ndarray):
            raise ValueError('Array should be numpy array')
        if len(datain.shape) == 2 or len(datain.shape) == 3:
            pass
        else:
            raise ValueError('Input array must be 2 or 3 dimensional')
    elif datatype == 'dict':
        if not isinstance(datain, dict):
            raise ValueError('Contact should be dictionary')
        if 'contacts' not in datain:
            raise ValueError('Key \'contacts\' should be in dictionary')
    elif datatype == 'df':
        if not isinstance(datain, pd.DataFrame):
            raise ValueError('Input should be Pandas Dataframe')
        if ('i' and 'j' and 't') not in datain:
            raise ValueError('Columns must be \'i\' \'j\' and \'t\'')
    else:
        raise ValueError('Unknown datatype')


def df_drop_ij_duplicates(df):
    """
    """
    df['ij'] = list(map(lambda x: tuple(sorted(x)), list(
        zip(*[df['i'].values, df['j'].values]))))
    df.drop_duplicates(['ij', 't'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.drop('ij', inplace=True, axis=1)
    return df
