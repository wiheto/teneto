import pandas as pd
import numpy as np
import teneto 
import inspect

class TemporalNetwork:

    def __init__(self, N=None, T=None, nettype=None, from_array=None, from_dict=None, from_edgelist=None, timetype=None): 
        # Check inputs 
        if nettype: 
            if nettype not in ['bu','bd','wu','wd']:
                raise ValueError('Nettype string must be: \'bu\', \'bd\', \'wu\' or \'wd\' for binary, weighted, undirected and directed.')

        if (from_array is not None and from_dict is not None) or (from_edgelist is not None and from_dict is not None) or (from_array is not None and from_edgelist is not None):
            raise ValueError('Cannot import from two sources at once.')

        if from_array is not None: 
            self._check_input_edgelist(from_array, 'array')
    
        if from_dict is not None: 
            self._check_input_edgelist(from_dict, 'dict')

        if from_edgelist is not None: 
            self._check_input_edgelist(from_edgelist, 'edgelist')

        if N: 
            if isinstance(N, int):
                raise ValueError('Number of nodes must be an interger')

        if T: 
            if isinstance(T, int):
                raise ValueError('Number of time-points must be an interger')

        if timetype: 
            if timetype not in ['discrete', 'continuous']:  
                raise ValueError('timetype must be \'discrete\' or \'continuous\'')
            self.timetype = timetype

        # Input
        if from_edgelist is not None: 
            if len(from_edgelist[0]) == 4: 
                colnames = ['i','j','t','weight']
            elif len(from_edgelist[0]) == 3: 
                colnames = ['i','j','t']
            self.network = pd.DataFrame(from_edgelist, columns=colnames) 
        elif from_array is not None: 
            uvals = np.unique(from_array)
            if len(uvals) == 2 and 1 in uvals and 0 in uvals: 
                i,j,t = np.where(from_array == 1)
                self.network = pd.DataFrame(data={'i': i, 'j': j, 't': t}) 
            else: 
                i,j,t = np.where(from_array != 0)
                w = from_array[from_array!=0]
                self.network = pd.DataFrame(data={'i': i, 'j': j, 't': t, 'weight': w}) 
        elif from_dict is not None: 
            self.network = pd.DataFrame(from_dict['contacts'], columns=['i', 'j', 't'])
            if 'values' in from_dict: 
                self.network['weight'] = from_dict['values']            

        if not nettype:
            print('No network type set: assuming it to be undirected, set nettype if directed') 
            if self.network.shape[-1] == 4:
                nettype = 'wu'
            elif self.network.shape[-1] == 3:
                nettype = 'bu'

        if not hasattr(self,'network'):
            if nettype[0] == 'w':
                colnames = ['i','j','t','weight']
            else:
                colnames = ['i','j','t']                
            self.network = pd.DataFrame(columns=colnames)
        
        self._calc_netshape()
        self.nettype = nettype
        if self.nettype[1] == 'u':
            self._drop_duplicate_ij()
        
    def _drop_duplicate_ij(self): 
        self.network['ij'] = list(map(lambda x: tuple(sorted(x)),list(zip(*[self.network['i'].values, self.network['j'].values]))))
        self.network.drop_duplicates('ij', inplace=True)
        self.network.reset_index(inplace=True, drop=True)
        self.network.drop('ij', inplace=True, axis=1)

    def _calc_netshape(self):
        if len(self.network) == 0: 
            self.netshape = (0,0)
        else: 
            N = self.network[['i','j']].max().max()
            T = self.network['t'].max()
            self.netshape = (N,T)

    def _check_input_edgelist(self, datain, datatype):
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
        else:
            raise ValueError('Unknown datatype')

    def add_edge(self, edgelist): 
        if not isinstance(edgelist[0], list): 
            edgelist = [edgelist]
        self._check_input_edgelist(edgelist, 'edgelist')
        if len(edgelist[0]) == 4: 
            colnames = ['i','j','t','weight']
        elif len(edgelist[0]) == 3: 
            colnames = ['i','j','t']
        newedges = pd.DataFrame(edgelist, columns=colnames) 
        self.network = pd.concat([self.network, newedges], ignore_index=True, sort=True)
        self._calc_netshape()
        if self.nettype[1] == 'u':
            self._drop_duplicate_ij()

    def drop_edge(self, edgelist): 
        if not isinstance(edgelist[0], list): 
            edgelist = [edgelist]
        self._check_input_edgelist(edgelist, 'edgelist')
        for e in edgelist: 
            idx = self.network[(self.network['i'] == e[0]) & (self.network['j'] == e[1]) & (self.network['t'] == e[2])].index
            self.network.drop(idx, inplace=True)
        self.network.reset_index(inplace=True, drop=True)

    def calc_networkmeasure(self, networkmeasure, **measureparams): 
        availablemeasures = [f for f in dir(teneto.networkmeasures) if not f.startswith('__')]
        if networkmeasure not in availablemeasures: 
            raise ValueError('Unknown network measure. Available network measures are: ' + ', '.join(availablemeasures))
        funs = inspect.getmembers(teneto.networkmeasures)
        funs={m[0]:m[1] for m in funs if not m[0].startswith('__')}
        measure = funs[networkmeasure](self,**measureparams)
        return measure

    def to_graphlet(self):

        idx = np.array(list(map(list, self.network.values)))
        G = np.zeros([self.netshape[0] + 1, self.netshape[0] + 1, self.netshape[1] + 1])
        if idx.shape[1] == 3:
            G[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        elif idx.shape[1] == 4:
            weights = idx[:,3]
            idx = np.array(idx[:,:3], dtype=int)
            G[idx[:, 0], idx[:, 1], idx[:, 2]] = weights
        if self.nettype[-1] == 'u': 
            G = G + G.transpose([1,0,2])
        return G