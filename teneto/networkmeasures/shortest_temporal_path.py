
import numpy as np
from teneto.utils import process_input
import itertools
import pandas as pd


def seqpath_to_path(pairseq, source):
    # seq must be a path sequence (i.e. possible paths per timepoint)
    # convert the sequence of pairs to a n x 2 array
    pairrows = np.reshape(pairseq, [int(len(pairseq)/2), 2])
    queue = [(0, [0])]
    # if source is in the first tuple, return
    if source in pairrows[0]:
        yield [pairrows[0].tolist()]
    while queue:
        # Set the queue
        (node, path) = queue.pop(0)
        # Get all remaining possible paths in sequence
        iterset = set(np.where((pairrows == pairrows[node, 0]) | (
            pairrows == pairrows[node, 1]))[0]) - set(range(node+1))
        for nextset in iterset:
            if source in pairrows[nextset]:
                yield list(reversed(pairrows[path + [nextset]].tolist()))
            else:
                queue.append((nextset, path + [nextset]))


def shortest_path_from_pairseq(pairseq, source):
    try:
        return next(seqpath_to_path(pairseq, source))
    except StopIteration:
        return None


def shortest_temporal_path(tnet, steps_per_t='all', i=None, j=None, it=None, minimise='temporal_distance'):
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

    minimise : str
        Can be "temporal_distance", returns the path that has the smallest temporal distance.
        It is possible there can be a path that is a smaller
        topological distance (this option currently not available).

    Returns
    -------------------
    paths : pandas df
        Dataframe consisting of information about all the paths found.

    Notes
    ---------------

    The shortest temporal path calculates the temporal and topological distance there to be a path between nodes.

    The argument steps_per_t allows for multiple nodes to be travelled per time-point.

    Topological distance is the number of edges that are travelled. Temporal distance is the number of time-points.

    This function returns the path that is the shortest temporal distance away.

    Examples
    --------

    Let us start by creating a small network.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import teneto
    >>> G = np.zeros([4, 4, 3])
    >>> G[0, 1, [0, 2]] = 1
    >>> G[0, 3, [2]] = 1
    >>> G[1, 2, [1]] = 1
    >>> G[2, 3, [1]] = 1

    Let us look at this network to see what is there.

    >>> fig, ax = plt.subplots(1)
    >>> ax = teneto.plot.slice_plot(G, ax, nodelabels=[0,1,2,3], timelabels=[0,1,2], cmap='Set2')
    >>> plt.tight_layout()
    >>> fig.show()

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        import teneto
        G = np.zeros([4, 4, 3])
        G[0, 1, [0, 2]] = 1
        G[0, 3, [2]] = 1
        G[1, 2, [1]] = 1
        G[2, 3, [1]] = 1
        fig,ax = plt.subplots(1)
        teneto.plot.slice_plot(G,ax,nodelabels=[0,1,2,3],timelabels=[0,1,2],cmap='Set2')
        plt.tight_layout()
        fig.show()

    Here we can visualize what the shortest paths are. Let us start by starting at
    node 0 we want to find the path to node 3, starting at time 0. To do this we write:

    >>> sp = teneto.networkmeasures.shortest_temporal_path(G, i=0, j=3, it=0)
    >>> sp['temporal-distance']
    0    2
    Name: temporal-distance, dtype: int64
    >>> sp['topological-distance']
    0    3
    Name: topological-distance, dtype: int64
    >>> sp['path includes']
    0    [[0, 1], [1, 2], [2, 3]]
    Name: path includes, dtype: object

    Here we see that the shortest path takes 3 steps (topological distance of 3) at 2 time points.

    It starts by going from node 0 to 1 at t=0, then 1 to 2 and 2 to 3 at t=1. We can see all the nodes
    that were travelled in the "path includes" list.

    In the above example, it was possible to traverse multiple edges at a single time-point.
    It is possible to restrain that by setting the steps_per_t argument

    >>> sp = teneto.networkmeasures.shortest_temporal_path(G, i=0, j=3, it=0, steps_per_t=1)
    >>> sp['temporal-distance']
    0    3
    Name: temporal-distance, dtype: int64
    >>> sp['topological-distance']
    0    1
    Name: topological-distance, dtype: int64
    >>> sp['path includes']
    0    [[0, 3]]
    Name: path includes, dtype: object

    Here we see that the path is now only one edge, 0 to 3 at t=2. The quicker path is no longer possible.

    """

    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')

    # If i, j or it are inputs, process them
    if i is None:
        source_nodes = np.arange(tnet.netshape[0])
    elif isinstance(i, int):
        source_nodes = [i]
    elif isinstance(i, list):
        source_nodes = i
    else:
        raise ValueError('Unknown i input. Should be None, int or list')
    if j is None:
        target_nodes = np.arange(tnet.netshape[0])
    elif isinstance(j, int):
        target_nodes = [j]
    elif isinstance(j, list):
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
                            network = tnet.get_network_when(ij=list(ij), t=t)
                        elif tnet.nettype[1] == 'd':
                            network = tnet.get_network_when(i=list(ij), t=t)
                        new_nodes = network[['i', 'j']].values
                        if len(new_nodes) != 0:
                            pairs.append(new_nodes.tolist())
                        new_nodes = new_nodes.flatten()
                        ij = np.hstack([ij, new_nodes])
                        ij = np.unique(ij)
                        if minimise == 'temporal_distance' and target in ij:
                            stop = 1
                        elif minimise == 'topology' and t == tnet.netshape[1] and target in ij:
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
                        # elif sourcei < 2 and target in a[:2]:
                        #    pl = 2

                    paths.append([source, target, tstart, t-tstart, pl, path])

    paths = pd.DataFrame(data=paths, columns=[
                         'from', 'to', 't_start', 'temporal-distance', 'topological-distance', 'path includes'])
    return paths
