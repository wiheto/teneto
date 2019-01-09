
import numpy as np
from teneto.utils import process_input
import itertools 
import pandas as pd 

def seqpath_to_path(pairseq, source):
    # seq must be a path sequence (i.e. possible paths per timepoint)
    # convert the sequence of pairs to a n x 2 array
    pairrows = np.reshape(pairseq,[int(len(pairseq)/2), 2])
    queue = [(0, [0])]
    # if source is in the first tuple, return
    if source in pairrows[0]:
        yield [pairrows[0].tolist()]
    while queue:
        # Set the queue
        (node, path) = queue.pop(0)
        # Get all remaining possible paths in sequence
        iterset = set(np.where((pairrows==pairrows[node,0]) | (pairrows==pairrows[node,1]))[0]) - set(range(node+1))
        for next in iterset:
            if source in pairrows[next]:
                yield list(reversed(pairrows[path + [next]].tolist()))
            else:
                queue.append((next, path + [next]))

def shortest_path_from_pairseq(pairseq, source):
    try:
        return next(seqpath_to_path(pairseq, source))
    except StopIteration:
        return None

def shortest_temporal_path(tnet, steps_per_t='all', i=None, j=None, it=None, minimise='time'):
    """ 
    Shortest temporal path

    Parameters
    --------------

    tnet : tnet obj, array or dict
        input network. nettype: bu, bd. 

    steps_per_t : int or str
        If str, should be 'all'. 
        How many edges can be travelled during a single time-point. 

    i : list 
        List of node indicies to restrict analysis. These are nodes the paths start from. Default is all nodes.

    j : list
        List of node indicies to restrict analysis. There are nodes the paths end on.  Default is all nodes.

    it : list
        List of starting time-point indicies to restrict anlaysis. Default is all timepoints.

    Returns 

    paths : pandas df 
        Dataframe consisting of information about all the paths found. 
    """

    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')

    # If i, j or it are inputs, process them
    if i is None: 
        source_nodes = np.arange(tnet.netshape[0])
    elif isinstance(i,int): 
        source_nodes = [i]
    elif isinstance(i,list): 
        source_nodes = i    
    else: 
        raise ValueError('Unknown i input. Should be None, int or list')
    if j is None: 
        target_nodes = np.arange(tnet.netshape[0])
    elif isinstance(j, int): 
        target_nodes = [j]
    elif isinstance(j,list): 
        target_nodes = j    
    else: 
        raise ValueError('Unknown j input. Should be None, int or list')
    if it is None: 
        time_points = np.arange(tnet.netshape[1])
    elif isinstance(it, int): 
        time_points = [it]
    elif isinstance(it, list): 
        time_points = it    
    else: 
        raise ValueError('Unknown t input. Should be None, int or list')

    # Two step process. 
    # First, get what the network can reach per timepoint. 
    # Second, check all possible sequences of what the network can reach for the shortest sequence.  
    paths = []
    for source in source_nodes:
        for target in target_nodes:
            if target == source: 
                pass
            else:
                for tstart in time_points:
                    # Part 1 starts here
                    ij = [source] 
                    t = tstart 
                    step = 1
                    lenij = 1
                    pairs = []
                    stop = 0
                    while stop == 0:
                        # Only select i if directed, ij if undirected. 
                        if tnet.nettype[1] == 'u': 
                            network = tnet.get_network_when(ij=list(ij),t=t)
                        elif tnet.nettype[1] == 'd': 
                            network = tnet.get_network_when(i=list(ij),t=t)
                        new_nodes = network[['i','j']].values
                        if len(new_nodes) != 0:
                            pairs.append(new_nodes.tolist())                 
                        new_nodes = new_nodes.flatten() 
                        ij = np.hstack([ij, new_nodes])
                        ij = np.unique(ij)
                        if minimise == 'time' and target in ij: 
                            stop = 1
                        elif minimise == 'topology' and t == tnet.netshape[1]  and target in ij:
                            stop = 1
                        elif t == tnet.netshape[1]:
                            t = np.nan
                            ij = [target]
                            stop = 1
                        else:
                            if len(ij) == lenij: 
                                t += 1
                                step = 1
                            elif steps_per_t == 'all':
                                pass
                            elif step < steps_per_t: 
                                step += 1
                            else: 
                                t += 1
                                step = 1
                        lenij = len(ij)
                    # correct t for return
                    t += 1
                    # Path 2 starts here
                    path = np.nan
                    pl = np.nan
                    for n in itertools.product(*reversed(pairs)): 
                        a = np.array(n).flatten()
                        if source not in a or target not in a: 
                            pass
                        else:
                            pathtmp = shortest_path_from_pairseq(a, source)
                            if pathtmp:
                                if not isinstance(path, list): 
                                    path = pathtmp
                                    pl = len(path)
                                elif len(pathtmp) < pl:  
                                    path = pathtmp
                                    pl = len(path)
                                elif len(pathtmp) == pl: 
                                    if isinstance(path[0][0], list):
                                        if pathtmp in path: 
                                            pass
                                        else:
                                            path.append(pathtmp)
                                    else: 
                                        if path == pathtmp:
                                            pass
                                        else:
                                            path = [path, pathtmp]
                        #elif sourcei < 2 and target in a[:2]:
                        #    pl = 2

                    paths.append([source,target,tstart,t-tstart,pl,path])

    paths = pd.DataFrame(data=paths, columns=['from', 'to', 't_start', 'temporal-distance', 'topological-distance', 'path includes'])
    return paths


