import pandas as pd
import numpy as np
import teneto 
import inspect
import matplotlib.pyplot as plt
import pickle

class TemporalNetwork:

    def __init__(self, N=None, T=None, nettype=None, from_df=None, from_array=None, from_dict=None, from_edgelist=None, timetype=None, diagonal=False,
                timeunit=None, desc=None, starttime=None, nodelabels=None, timelabels=None): 
        # Check inputs 
        if nettype: 
            if nettype not in ['bu','bd','wu','wd']:
                raise ValueError('Nettype string must be: \'bu\', \'bd\', \'wu\' or \'wd\' for binary, weighted, undirected and directed.')

        inputvars = locals()
        if sum([1 for n in  inputvars.keys() if 'from' in n and inputvars[n] is not None]) > 1:
            raise ValueError('Cannot import from two sources at once.')

        if from_array is not None: 
            self._check_input(from_array, 'array')
    
        if from_dict is not None: 
            self._check_input(from_dict, 'dict')

        if from_edgelist is not None: 
            self._check_input(from_edgelist, 'edgelist')

        if N: 
            if not isinstance(N, int):
                raise ValueError('Number of nodes must be an interger')

        if T: 
            if not isinstance(T, int):
                raise ValueError('Number of time-points must be an interger')

        if timetype: 
            if timetype not in ['discrete', 'continuous']:  
                raise ValueError('timetype must be \'discrete\' or \'continuous\'')
            self.timetype = timetype

        self.diagonal = diagonal

        #todo - add checks that labels are ok
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
            self.network_from_array(from_array)
        elif from_dict is not None: 
            self.network_from_dict(from_dict)

        if not hasattr(self,'network'):
            if nettype: 
                if nettype[0] == 'w':
                    colnames = ['i','j','t','weight']
                else:
                    colnames = ['i','j','t']                
            else:
                colnames = ['i','j','t']
            self.network = pd.DataFrame(columns=colnames)
                
        # Update df 
        self._calc_netshape()
        if not self.diagonal:
            self._drop_diagonal()
        if nettype:
            if nettype[1] == 'u':
                self._drop_duplicate_ij()

    def _set_nettype(self):
        # Only run if not manually set and network values exist
        if not hasattr(self,'nettype') and len(self.network) > 0:
            # Predefine
            nettype = 'xx'
            # Then check if weighted
            if 'weight' in self.network.columns: 
                wb = 'w'
            else: 
                wb = 'b'
            # Would be good to see if there was a way to this without going to array.
            self.nettype = 'xu'
            G1 = self.to_array()
            self.nettype = 'xd'
            G2 = self.to_array()
            if np.all(G1==G2): 
                ud = 'u'
            else:
                ud= 'd'
            self.nettype = wb + ud
        
    def network_from_array(self, array):
        """impo
        Defines a network from an array. 

        Parameters
        ----------
        array : array 
            3D numpy array. 
        """
        self._check_input(array, 'array')
        uvals = np.unique(array)
        if len(uvals) == 2 and 1 in uvals and 0 in uvals: 
            i,j,t = np.where(array == 1)
            self.network = pd.DataFrame(data={'i': i, 'j': j, 't': t}) 
        else: 
            i,j,t = np.where(array != 0)
            w = array[array!=0]
            self.network = pd.DataFrame(data={'i': i, 'j': j, 't': t, 'weight': w}) 
        self.netshape = array.shape
        self._update_network()
    
    def _update_network(self):
        self._calc_netshape()
        self._set_nettype()
        if self.nettype:
            if self.nettype[1] == 'u':
                self._drop_duplicate_ij()

            
    def network_from_df(self, df):
        """
        Defines a network from an array. 

        Parameters
        ----------
        array : array 
            Pandas dataframe. Should have columns: \'i\', \'j\', \'t\' where i and j are node indicies and t is the temporal index. 
            If weighted, should also include \'weight\'. Each row is an edge.  
        """
        self._check_input(df, 'df')
        self.network = df 
        self._update_network()

    
    def network_from_edgelist(self, edgelist):
        """
        Defines a network from an array. 

        Parameters
        ----------
        edgelist : list of lists.  
            A list of lists which are 3 or 4 in length. For binary networks each sublist should be [i, j ,t] where i and j are node indicies and t is the temporal index.
            For weighted networks each sublist should be [i, j, t, weight].
        """
        self._check_input(edgelist, 'edgelist')
        if len(edgelist[0]) == 4: 
            colnames = ['i','j','t','weight']
        elif len(edgelist[0]) == 3: 
            colnames = ['i','j','t']
        self.network = pd.DataFrame(edgelist, columns=colnames) 
        self._update_network()
    
    def network_from_dict(self, contact):
    
        self._check_input(contact, 'dict')
        self.network = pd.DataFrame(contact['contacts'], columns=['i', 'j', 't'])
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
        """
        Drops duplicate entries from the network dataframe. 
        """
        self.network['ij'] = list(map(lambda x: tuple(sorted(x)),list(zip(*[self.network['i'].values, self.network['j'].values]))))
        self.network.drop_duplicates(['ij','t'], inplace=True)
        self.network.reset_index(inplace=True, drop=True)
        self.network.drop('ij', inplace=True, axis=1)

    def _drop_diagonal(self): 
        """
        Drops self-contacts from the network dataframe. 
        """
        self.network = self.network.where(self.network['i'] != self.network['j']).dropna()
        self.network.reset_index(inplace=True, drop=True)

    def _calc_netshape(self):
        if len(self.network) == 0: 
            self.netshape = (0,0)
        else: 
            N = self.network[['i','j']].max(axis=1).max()+1
            T = self.network['t'].max()+1
            self.netshape = (int(N),int(T))

    def _check_input(self, datain, datatype):
        if datatype == 'edgelist': 
            if not isinstance(datain, list): 
                raise ValueError('edgelist should be list')
            if all([len(e)==3 for e in datain]) or all([len(e)==4 for e in datain]):
                pass
            else: 
                raise ValueError('Each member in edgelist should all be a list of length 3 (i,j,t) or 4 (i,j,t,w)')
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

    def add_edge(self, edgelist): 
        if not isinstance(edgelist[0], list): 
            edgelist = [edgelist]
        self._check_input(edgelist, 'edgelist')
        if len(edgelist[0]) == 4: 
            colnames = ['i','j','t','weight']
        elif len(edgelist[0]) == 3: 
            colnames = ['i','j','t']
        newedges = pd.DataFrame(edgelist, columns=colnames) 
        self.network = pd.concat([self.network, newedges], ignore_index=True, sort=True)
        self._update_network()

    def drop_edge(self, edgelist): 
        if not isinstance(edgelist[0], list): 
            edgelist = [edgelist]
        self._check_input(edgelist, 'edgelist')
        for e in edgelist: 
            idx = self.network[(self.network['i'] == e[0]) & (self.network['j'] == e[1]) & (self.network['t'] == e[2])].index
            self.network.drop(idx, inplace=True)
        self.network.reset_index(inplace=True, drop=True)
        self._update_network()

    def calc_networkmeasure(self, networkmeasure, **measureparams): 
        availablemeasures = [f for f in dir(teneto.networkmeasures) if not f.startswith('__')]
        if networkmeasure not in availablemeasures: 
            raise ValueError('Unknown network measure. Available network measures are: ' + ', '.join(availablemeasures))
        funs = inspect.getmembers(teneto.networkmeasures)
        funs={m[0]:m[1] for m in funs if not m[0].startswith('__')}
        measure = funs[networkmeasure](self,**measureparams)
        return measure

    def generatenetwork(self, networktype, **networkparams): 
        availabletypes = [f for f in dir(teneto.generatenetwork) if not f.startswith('__')]
        if networktype not in availabletypes: 
            raise ValueError('Unknown network measure. Available networks to generate are: ' + ', '.join(availabletypes))
        funs = inspect.getmembers(teneto.generatenetwork)
        funs={m[0]:m[1] for m in funs if not m[0].startswith('__')}
        network = funs[networktype](**networkparams)
        self.network_from_array(network)
        if self.nettype[1] == 'u':
            self._drop_duplicate_ij()

    def plot(self, plottype, ax=None, **plotparams): 
        if 'nodelabels' not in plotparams and self.nodelabels:
            plotparams['nodelabels'] = self.nodelabels 
        if 'timeunit' not in plotparams and self.timeunit:
            plotparams['timeunit'] = self.timeunit 
        if 'timelabels' not in plotparams and self.timelabels:
            plotparams['timelabels'] = self.timelabels 
        availabletypes = [f for f in dir(teneto.plot) if not f.startswith('__')]
        if plottype not in availabletypes: 
            raise ValueError('Unknown network measure. Available plotting functions are: ' + ', '.join(availabletypes))
        funs = inspect.getmembers(teneto.plot)
        funs={m[0]:m[1] for m in funs if not m[0].startswith('__')}
        if not ax: 
            _, ax = plt.subplots(1)
        ax = funs[plottype](self.to_array(), ax=ax, **plotparams)
        return ax

    def to_array(self):
        if len(self.network) > 0: 
            idx = np.array(list(map(list, self.network.values)))
            G = np.zeros([self.netshape[0], self.netshape[0], self.netshape[1]])
            if idx.shape[1] == 3:
                if self.nettype[-1] == 'u': 
                    idx = np.vstack([idx,idx[:,[1,0,2]]])
                idx = idx.astype(int)
                G[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
            elif idx.shape[1] == 4:
                if self.nettype[-1] == 'u': 
                    idx = np.vstack([idx,idx[:,[1,0,2,3]]])
                weights = idx[:,3]
                idx = np.array(idx[:,:3], dtype=int)
                G[idx[:, 0], idx[:, 1], idx[:, 2]] = weights
        else: 
            G = np.zeros([self.netshape[0],self.netshape[0],self.netshape[1]])
        return G

    def save_aspickle(self, fname):
        if fname[-4:] != '.pkl':
            fname += '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def get_network_when(self, i=None, j=None, t=None, ij=None, logic='and', copy=False, asarray=False): 
        """
        Returns subset of dataframe that matches index

        Parameters
        ----------
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
            default False. If True, returns a copy of the dataframe. 
        asarray : bool 
            default False. If True, returns the list of edges as an array. 

        Returns
        -------
        df : pandas dataframe
            Unless asarray are set to true. 
        """
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
        if i is not None and j is not None and t is not None and logic == 'and': 
            df = self.network[(self.network['i'].isin(i)) & (self.network['j'].isin(j)) & (self.network['t'].isin(t))]
        elif ij is not None and t is not None and logic == 'and': 
            df = self.network[((self.network['i'].isin(ij)) | (self.network['j'].isin(ij))) & (self.network['t'].isin(t))]
        elif ij is not None and t is not None and logic == 'or': 
            df = self.network[((self.network['i'].isin(ij)) | (self.network['j'].isin(ij))) | (self.network['t'].isin(t))]
        elif i is not None and j is not None and logic == 'and': 
            df = self.network[(self.network['i'].isin(i)) & (self.network['j'].isin(j))]        
        elif i is not None and t is not None and logic == 'and': 
            df = self.network[(self.network['i'].isin(i)) & (self.network['t'].isin(t))]        
        elif j is not None and t is not None and logic == 'and': 
            df = self.network[(self.network['j'].isin(j)) & (self.network['t'].isin(t))]   
        elif i is not None and j is not None and t is not None and logic == 'or': 
            df = self.network[(self.network['i'].isin(i)) | (self.network['j'].isin(j)) | (self.network['t'].isin(t))]
        elif i is not None and j is not None and logic == 'or': 
            df = self.network[(self.network['i'].isin(i)) | (self.network['j'].isin(j))]        
        elif i is not None and t is not None and logic == 'or': 
            df = self.network[(self.network['i'].isin(i)) | (self.network['t'].isin(t))]        
        elif j is not None and t is not None and logic == 'or': 
            df = self.network[(self.network['j'].isin(j)) | (self.network['t'].isin(t))]        
        elif i is not None:
            df = self.network[self.network['i'].isin(i)]
        elif j is not None:
            df = self.network[self.network['j'].isin(j)]
        elif t is not None:
            df = self.network[self.network['t'].isin(t)]
        elif ij is not None:
            df = self.network[(self.network['i'].isin(ij)) | (self.network['j'].isin(ij))]
        if copy: 
            df = df.copy()
        if asarray: 
            df = df.values
        return df
