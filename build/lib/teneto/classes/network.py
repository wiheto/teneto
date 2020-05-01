"""The TemporalNetwork class in teneto is a way of representing network objects."""

import pandas as pd
import numpy as np
import teneto
import inspect
import matplotlib.pyplot as plt


class TemporalNetwork:
    """
    A class for temporal networks.

    This class allows to call different teneto functions within the class and store the network representation.

    Parameters
    ============

    N : int
        number of nodes in network
    T : int
        number of time-points in network
    nettype : str
        description of network. Can be: bu, bd, wu, wd where the letters stand for binary, weighted, undirected and directed.
        Default is weighted and undirected.
    from_df : pandas df
        input data frame with i,j,t,[weight] columns
    from_array : array
        input data from an array with dimesnions node,node,time
    from_dict : dict
        input data is a contact sequence dictionary.
    from_edgelist : list
        input data is a list of lists where each item in main list consists of [i,j,t,[weight]].
    timetype : str
        discrete or continuous
    diagonal : bool
        if the diagonal should be included in the edge list.
    timeunit : str
        string (used in plots)
    desc : str
        string to describe network.
    startime : int
        integer represents time of first index.
    nodelabels : list
        list of labels for naming the nodes
    timelabels : list
        list of labels for time-points
    hdf5 : bool
        if true, pandas dataframe is stored and queried as a h5 file.
    hdf5path : str
        Where the h5 files is saved if hdf5 is True. If left unset, the default is ./teneto_temporalnetwork.h5
    forcesparse : bool
        If importing array, and over 25% edges are present, a dense matrix is created. Can force it to be sparse by making this true.
    """

    def __init__(self, N=None, T=None, nettype=None, from_df=None, from_array=None, from_dict=None, from_edgelist=None, timetype=None, diagonal=False,
                 timeunit=None, desc=None, starttime=None, nodelabels=None, timelabels=None, hdf5=False, hdf5path=None, forcesparse=False):
        # Check inputs
        if nettype:
            if nettype not in ['bu', 'bd', 'wu', 'wd']:
                raise ValueError(
                    'Nettype string must be: \'bu\', \'bd\', \'wu\' or \'wd\' for binary, weighted, undirected and directed.')

        inputvars = locals()
        if sum([1 for n in inputvars.keys() if 'from' in n and inputvars[n] is not None]) > 1:
            raise ValueError('Cannot import from two sources at once.')

        if from_array is not None:
            teneto.utils.check_TemporalNetwork_input(from_array, 'array')

        if from_dict is not None:
            teneto.utils.check_TemporalNetwork_input(from_dict, 'dict')

        if from_edgelist is not None:
            teneto.utils.check_TemporalNetwork_input(from_edgelist, 'edgelist')

        if N:
            if not isinstance(N, int):
                raise ValueError('Number of nodes must be an interger')

        if T:
            if not isinstance(T, int):
                raise ValueError('Number of time-points must be an interger')

        if N is None:
            self.N = 0
        else:
            self.N = int(N)
        if T is None:
            self.T = 0
        else:
            self.T = int(T)

        if timetype:
            if timetype not in ['discrete', 'continuous']:
                raise ValueError(
                    'timetype must be \'discrete\' or \'continuous\'')
            self.timetype = timetype

        if hdf5:
            if hdf5path is None:
                hdf5path = './teneto_temporalnetwork.h5'
            if hdf5path[:-3:] == '.h5':
                hdf5path = hdf5path[:-3]

        self.diagonal = diagonal
        self.sparse = True
        # todo - add checks that labels are ok
        if nodelabels:
            self.nodelabels = nodelabels
        else:
            self.nodelabels = None

        if timelabels:
            self.timelabels = timelabels
        else:
            self.timelabels = None

        if timeunit:
            self.timeunit = timeunit
        else:
            self.timeunit = None

        if starttime:
            self.starttime = starttime
        else:
            self.starttime = 0

        if desc:
            self.desc = desc
        else:
            self.desc = None

        if nettype:
            self.nettype = nettype

        # Input
        if from_df is not None:
            self.network_from_df(from_df)
        if from_edgelist is not None:
            self.network_from_edgelist(from_edgelist)
        elif from_array is not None:
            self.network_from_array(from_array, forcesparse=forcesparse)
        elif from_dict is not None:
            self.network_from_dict(from_dict)

        if not hasattr(self, 'network'):
            if nettype:
                if nettype[0] == 'w':
                    colnames = ['i', 'j', 't', 'weight']
                else:
                    colnames = ['i', 'j', 't']
            else:
                colnames = ['i', 'j', 't']
            self.network = pd.DataFrame(columns=colnames)

        # Update df
        self._calc_netshape()
        if not self.diagonal:
            self._drop_diagonal()
        if nettype and self.sparse:
            if nettype[1] == 'u':
                self._drop_duplicate_ij()

        self.hdf5 = False
        if hdf5:
            self.hdf5_setup(hdf5path)

    def _set_nettype(self):
        """Helper function that sets the network type"""
        # Only run if not manually set and network values exist
        if not hasattr(self, 'nettype') and len(self.network) > 0:
            # Then check if weighted
            if 'weight' in self.network.columns:
                wb = 'w'
            else:
                wb = 'b'
            # Would be good to see if there was a way to this without going to array.
            self.nettype = 'xu'
            G1 = teneto.utils.df_to_array(
                self.network, self.netshape, self.nettype)
            self.nettype = 'xd'
            G2 = teneto.utils.df_to_array(
                self.network, self.netshape, self.nettype)
            if np.all(G1 == G2):
                ud = 'u'
            else:
                ud = 'd'
            self.nettype = wb + ud

    def network_from_array(self, array, forcesparse=False):
        """
        Defines a network from an array.

        Parameters
        ----------
        array : array
            3D numpy array.
        forcespace : bool
            If true, will always make the array sparse (can be slow). If false, dense form will be kept
            if more than 25% of edges are present.
        """
        if len(array.shape) == 2:
            array = np.array(array, ndmin=3).transpose([1, 2, 0])
        teneto.utils.check_TemporalNetwork_input(array, 'array')
        if np.sum([array == 0]) > np.prod(array.shape)*0.75 or forcesparse:
            uvals = np.unique(array)
            if len(uvals) == 2 and 1 in uvals and 0 in uvals:
                i, j, t = np.where(array == 1)
                self.network = pd.DataFrame(data={'i': i, 'j': j, 't': t})
            else:
                i, j, t = np.where(array != 0)
                w = array[array != 0]
                self.network = pd.DataFrame(
                    data={'i': i, 'j': j, 't': t, 'weight': w})
            self._update_network()
        else:
            self.network = np.array(array)
            self.sparse = False
            self.nettype = teneto.utils.gen_nettype(self.network)
        self.N = int(array.shape[0])
        self.T = int(array.shape[-1])
        self.netshape = (self.N, self.T)

    def _update_network(self):
        """Helper function that updates the network info"""
        self._calc_netshape()
        self._set_nettype()
        if self.nettype:
            if self.nettype[1] == 'u':
                self._drop_duplicate_ij()
        self.network['i'] = self.network['i'].astype(int)
        self.network['j'] = self.network['j'].astype(int)

    def network_from_df(self, df):
        r"""
        Defines a network from an array.

        Parameters
        ----------
        array : array
            Pandas dataframe. Should have columns: \'i\', \'j\', \'t\' where i and j are node indicies and t is the temporal index.
            If weighted, should also include \'weight\'. Each row is an edge.
        """
        teneto.utils.check_TemporalNetwork_input(df, 'df')
        self.network = df
        self._update_network()

    def network_from_edgelist(self, edgelist):
        """
        Defines a network from an array.

        Parameters
        ----------
        edgelist : list of lists.
            A list of lists which are 3 or 4 in length.
            For binary networks each sublist should be [i, j ,t] where i and j are node indicies and t is the temporal index.
            For weighted networks each sublist should be [i, j, t, weight].
        """
        teneto.utils.check_TemporalNetwork_input(edgelist, 'edgelist')
        if len(edgelist[0]) == 4:
            colnames = ['i', 'j', 't', 'weight']
        elif len(edgelist[0]) == 3:
            colnames = ['i', 'j', 't']
        self.network = pd.DataFrame(edgelist, columns=colnames)
        self._update_network()

    def network_from_dict(self, contact):
        """
        """
        teneto.utils.check_TemporalNetwork_input(contact, 'dict')
        self.network = pd.DataFrame(
            contact['contacts'], columns=['i', 'j', 't'])
        if 'values' in contact:
            self.network['weight'] = contact['values']
        self.nettype = contact['nettype']
        self.starttime = contact['t0']
        self.netshape = contact['netshape']
        if contact['nodelabels']:
            self.nodelabels = contact['nodelabels']
        if contact['timeunit']:
            self.timeunit = contact['timeunit']

    def _drop_duplicate_ij(self):
        """Drops duplicate entries from the network dataframe."""
        self.network = teneto.utils.df_drop_ij_duplicates(self.network)

    def _drop_diagonal(self):
        """Drops self-contacts from the network dataframe."""
        if self.sparse:
            self.network = self.network.where(
                self.network['i'] != self.network['j']).dropna()
            self.network.reset_index(inplace=True, drop=True)
        else:
            self.network = teneto.utils.set_diagonal(self.network, 0)

    def _calc_netshape(self):
        """
        """
        if len(self.network) == 0:
            self.netshape = (0, 0)
        elif not self.sparse:
            N = int(self.network.shape[0])
            T = int(self.network.shape[-1])
            self.netshape = (N, T)
        else:
            N = self.network[['i', 'j']].max(axis=1).max()+1
            T = self.network['t'].max()+1
            if self.N > N:
                N = self.N
            else:
                self.N = int(N)
            if self.T > T:
                T = self.T
            else:
                self.T = int(T)
            self.netshape = (int(N), int(T))

    def add_edge(self, edgelist):
        """
        Adds an edge from network.

        Parameters
        ----------

        edgelist : list
            a list (or list of lists) containing the i,j and t indicies to be added. For weighted networks list should also contain a 'weight' key.

        Returns
        --------
            Updates TenetoBIDS.network dataframe with new edge
        """
        if not self.sparse:
            raise ValueError('Add edge not compatible with dense network')
        if not isinstance(edgelist[0], list):
            edgelist = [edgelist]
        teneto.utils.check_TemporalNetwork_input(edgelist, 'edgelist')
        if len(edgelist[0]) == 4:
            colnames = ['i', 'j', 't', 'weight']
        elif len(edgelist[0]) == 3:
            colnames = ['i', 'j', 't']
        if self.hdf5:
            with pd.HDFStore(self.network) as hdf:
                rows = hdf.get_storer('network').nrows
                hdf.append('network', pd.DataFrame(edgelist, columns=colnames, index=np.arange(
                    rows, rows+len(edgelist))), format='table', data_columns=True)
            edgelist = np.array(edgelist)
            if np.max(edgelist[:, :2]) > self.netshape[0]:
                self.netshape[0] = np.max(edgelist[:, :2])
            if np.max(edgelist[:, 2]) > self.netshape[1]:
                self.netshape[1] = np.max(edgelist[:, 2])
        else:
            newedges = pd.DataFrame(edgelist, columns=colnames)
            self.network = pd.concat(
                [self.network, newedges], ignore_index=True, sort=True)
            self._update_network()

    def drop_edge(self, edgelist):
        """
        Removes an edge from network.

        Parameters
        ----------

        edgelist : list
            a list (or list of lists) containing the i,j and t indicies to be removes.

        Returns
        --------
            Updates TenetoBIDS.network dataframe
        """
        if not isinstance(edgelist[0], list):
            edgelist = [edgelist]
        teneto.utils.check_TemporalNetwork_input(edgelist, 'edgelist')
        if self.hdf5:
            with pd.HDFStore(self.network) as hdf:
                for e in edgelist:
                    hdf.remove(
                        'network', 'i == ' + str(e[0]) + ' & ' + 'j == ' + str(e[1]) + ' & ' + 't == ' + str(e[2]))
            print('HDF5 delete warning. This will not reduce the size of the file.')
        else:
            for e in edgelist:
                idx = self.network[(self.network['i'] == e[0]) & (
                    self.network['j'] == e[1]) & (self.network['t'] == e[2])].index
                self.network.drop(idx, inplace=True)
            self.network.reset_index(inplace=True, drop=True)
            self._update_network()

    def calc_networkmeasure(self, networkmeasure, **measureparams):
        """
        Calculate network measure.

        Parameters
        -----------
        networkmeasure : str
            Function to call. Functions available are in teneto.networkmeasures

        measureparams : kwargs
            kwargs for teneto.networkmeasure.[networkmeasure]
        """
        availablemeasures = [f for f in dir(
            teneto.networkmeasures) if not f.startswith('__')]
        if networkmeasure not in availablemeasures:
            raise ValueError(
                'Unknown network measure. Available network measures are: ' + ', '.join(availablemeasures))
        funs = inspect.getmembers(teneto.networkmeasures)
        funs = {m[0]: m[1] for m in funs if not m[0].startswith('__')}
        measure = funs[networkmeasure](self, **measureparams)
        return measure

    def generatenetwork(self, networktype, **networkparams):
        """
        Generate a network

        Parameters
        -----------
        networktype : str
            Function to call. Functions available are in teneto.generatenetwork

        measureparams : kwargs
            kwargs for teneto.generatenetwork.[networktype]

        Returns
        --------
        TenetoBIDS.network is made with the generated network.
        """
        availabletypes = [f for f in dir(
            teneto.generatenetwork) if not f.startswith('__')]
        if networktype not in availabletypes:
            raise ValueError(
                'Unknown network measure. Available networks to generate are: ' + ', '.join(availabletypes))
        funs = inspect.getmembers(teneto.generatenetwork)
        funs = {m[0]: m[1] for m in funs if not m[0].startswith('__')}
        network = funs[networktype](**networkparams)
        self.network_from_array(network)
        if self.nettype[1] == 'u' and self.sparse == 'True':
            self._drop_duplicate_ij()

    def plot(self, plottype, ij=None, t=None, ax=None, **plotparams):
        """
        """
        if 'nodelabels' not in plotparams and self.nodelabels:
            plotparams['nodelabels'] = self.nodelabels
        if 'timeunit' not in plotparams and self.timeunit:
            plotparams['timeunit'] = self.timeunit
        if 'timelabels' not in plotparams and self.timelabels:
            plotparams['timelabels'] = self.timelabels
        availabletypes = [f for f in dir(
            teneto.plot) if not f.startswith('__')]
        if plottype not in availabletypes:
            raise ValueError(
                'Unknown network measure. Available plotting functions are: ' + ', '.join(availabletypes))
        funs = inspect.getmembers(teneto.plot)
        funs = {m[0]: m[1] for m in funs if not m[0].startswith('__')}
        if ij is None:
            ij = np.arange(self.netshape[0]).tolist()
        if t is None:
            t = np.arange(self.netshape[1]).tolist()
        if not ax:
            _, ax = plt.subplots(1)
        data_plot = teneto.utils.get_network_when(self, ij=ij, t=t)
        data_plot = teneto.utils.df_to_array(
            data_plot, self.netshape, self.nettype)
        ax = funs[plottype](data_plot, ax=ax, **plotparams)
        return ax

    def hdf5_setup(self, hdf5path):
        """
        """
        hdf = pd.HDFStore(hdf5path)
        hdf.put('network', self.network, format='table', data_columns=True)
        hdf.close()
        self.hdf5 = True
        self.network = hdf5path

    def get_network_when(self, **kwargs):
        """
        """
        return teneto.utils.get_network_when(self, **kwargs)

    def df_to_array(self):
        """
        """
        return teneto.utils.df_to_array(self.network, self.netshape, self.nettype)

    def binarize(self, threshold_type, threshold_level, **kwargs):
        """
        Binarizes the network.

        Parameters
        ----------

        threshold_type : str
            What type of thresholds to make binarization. Options: 'rdp', 'percent', 'magnitude'.

        threshold_level : str
            Paramter dependent on threshold type.
            If 'rdp', it is the delta (i.e. error allowed in compression).
            If 'percent', it is the percentage to keep (e.g. 0.1, means keep 10% of signal).
            If 'magnitude', it is the amplitude of signal to keep.

        See teneto.utils.binarize for kwarg arguments.

        Returns
        ---------
        Updates tnet.network to be binarized

        """
        gbin = teneto.utils.binarize(
            self.network, threshold_type, threshold_level, **kwargs)
        if self.sparse:
            gbin = teneto.utils.process_input(
                gbin, 'G', outputformat='TN', forcesparse=True)
            self.network = gbin.network
        else:
            self.network = gbin
        self.nettype = 'b' + self.nettype[1]
