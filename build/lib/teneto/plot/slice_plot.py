"""Draw a slice_graph"""
import numpy as np
from ..utils import check_input, graphlet2contact


def slice_plot(netin, ax, nodelabels=None, timelabels=None,
               communities=None, plotedgeweights=False, edgeweightscalar=1,
               timeunit='', linestyle='k-', cmap=None, nodesize=100,
               nodekwargs=None, edgekwargs=None):
    r"""
    Fuction draws "slice graph".

    A slice plot plots all the nodes per time point as a column
    with Bezier curvers connecting connected nodes.

    Parameters
    ----------

    netin : array, dict
        temporal network input (graphlet or contact)
    ax : matplotlib figure handles.
    nodelabels : list
        nodes labels. List of strings.
    timelabels : list
        labels of dimension Graph is expressed across. List of strings.
    communities : array
        array of size: (time) or (node,time). Nodes will be coloured accordingly.
    plotedgeweights : bool
        if True, edges will vary in size (default False)
    edgeweightscalar : int
        scalar to multiply all edges if tweaking is needed.
    timeunit : string
        unit time axis is in.
    linestyle : string
        line style of Bezier curves.
    nodesize : int
        size of nodes
    nodekwargs : dict
        any additional kwargs for matplotlib.plt.scatter for the nodes
    edgekwargs : dict
        any additional kwargs for matplotlib.plt.plots for the edges


    Returns
    ---------
    ax : axis handle of slice graph


    Examples
    ---------


    Create a network with some metadata

    >>> import numpy as np
    >>> import teneto
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(2017) # For reproduceability
    >>> N = 5 # Number of nodes
    >>> T = 10 # Number of timepoints
    >>> # Probability of edge activation
    >>> birth_rate = 0.2
    >>> death_rate = .9
    >>> # Add node names into the network and say time units are years, go 1 year per graphlet and startyear is 2007
    >>> cfg={}
    >>> cfg['Fs'] = 1
    >>> cfg['timeunit'] = 'Years'
    >>> cfg['t0'] = 2007 #First year in network
    >>> cfg['nodelabels'] = ['Ashley','Blake','Casey','Dylan','Elliot'] # Node names
    >>> #Generate network
    >>> C = teneto.generatenetwork.rand_binomial([N,T],[birth_rate, death_rate],'contact','bu',netinfo=cfg)

    Now this network can be plotted

    >>> fig,ax = plt.subplots(figsize=(10,3))
    >>> ax = teneto.plot.slice_plot(C, ax, cmap='Pastel2')
    >>> plt.tight_layout()
    >>> fig.show()

    .. plot::

        import numpy as np
        import teneto
        import matplotlib.pyplot as plt
        np.random.seed(2017) # For reproduceability
        N = 5 # Number of nodes
        T = 10 # Number of timepoints
        # Probability of edge activation
        birth_rate = 0.2
        death_rate = .9
        # Add node names into the network and say time units are years, go 1 year per graphlet and startyear is 2007
        cfg={}
        cfg['Fs'] = 1
        cfg['timeunit'] = 'Years'
        cfg['t0'] = 2007 #First year in network
        cfg['nodelabels'] = ['Ashley','Blake','Casey','Dylan','Elliot']
        #Generate network
        C = teneto.generatenetwork.rand_binomial([N,T],[birth_rate, death_rate],'contact','bu',netinfo=cfg)
        fig,ax = plt.subplots(figsize=(10,3))
        cmap = 'Pastel2'
        ax = teneto.plot.slice_plot(C,ax,cmap=cmap)
        plt.tight_layout()
        fig.show()

    """
    # Get input type (C or G)
    inputType = check_input(netin)
    # Convert C representation to G

    if inputType == 'G':
        netin = graphlet2contact(netin)
        inputType = 'C'
    edgelist = [tuple(np.array(e[0:2]) + e[2] * netin['netshape'][0])
                for e in netin['contacts']]

    if nodelabels is not None and len(nodelabels) == netin['netshape'][0]:
        pass
    elif nodelabels is not None and len(nodelabels) != netin['netshape'][0]:
        raise ValueError('specified node label length does not match netshape')
    elif nodelabels is None and netin['nodelabels'] == '':
        nodelabels = np.arange(1, netin['netshape'][0] + 1)
    else:
        nodelabels = netin['nodelabels']

    if timelabels is not None and len(timelabels) == netin['netshape'][-1]:
        pass
    elif timelabels is not None and len(timelabels) != netin['netshape'][-1]:
        raise ValueError('specified time label length does not match netshape')
    elif timelabels is None and str(netin['t0']) == '':
        timelabels = np.arange(1, netin['netshape'][-1] + 1)
    else:
        timelabels = np.arange(netin['t0'], netin['Fs'] *
                               netin['netshape'][-1] + netin['t0'], netin['Fs'])

    if timeunit is None:
        timeunit = netin['timeunit']

    timeNum = len(timelabels)
    nodeNum = len(nodelabels)
    posy = np.tile(list(range(0, nodeNum)), timeNum)
    posx = np.repeat(list(range(0, timeNum)), nodeNum)

    if nodekwargs is None:
        nodekwargs = {}
    if edgekwargs is None:
        edgekwargs = {}
    if cmap:
        nodekwargs['cmap'] = cmap
    if 'c' not in nodekwargs:
        nodekwargs['c'] = posy
    if communities is not None:
        # check if temporal or static
        if len(communities.shape) == 1:
            nodekwargs['c'] = np.tile(communities, timeNum)
        else:
            nodekwargs['c'] = communities.flatten(order='F')

    # plt.plot(points)
    # Draw Bezier vectors around egde positions
    for ei, edge in enumerate(edgelist):
        if plotedgeweights and netin['nettype'][0] == 'w':
            edgekwargs['linewidth'] = netin['values'][ei] * edgeweightscalar
        bvx, bvy = bezier_points(
            (posx[edge[0]], posy[edge[0]]), (posx[edge[1]], posy[edge[1]]), nodeNum, 20)
        ax.plot(bvx, bvy, linestyle, **edgekwargs)
    ax.set_yticks(range(0, len(nodelabels)))
    ax.set_xticks(range(0, len(timelabels)))
    ax.set_yticklabels(nodelabels)
    ax.set_xticklabels(timelabels)
    ax.grid()
    ax.set_frame_on(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([min(posx) - 1, max(posx) + 1])
    ax.set_ylim([min(posy) - 1, max(posy) + 1])
    ax.scatter(posx, posy, s=nodesize, zorder=10, **nodekwargs)
    if timeunit != '':
        timeunit = ' (' + timeunit + ')'
    ax.set_xlabel('Time' + timeunit)

    return ax


# Following 3 Function that draw vertical curved lines from around points.
# p1 nad p2 are start and end trupes (x,y coords) and pointN is the resolution of the points
# negxLim tries to restrain how far back along the x axis the bend can go.
def bezier_points(p1, p2, negxLim, pointN):
    ts = [t / pointN for t in range(pointN + 1)]
    d = p1[0] - (max(p1[1], p2[1]) - min(p1[1], p2[1])) / negxLim
    bezier = make_bezier([p1, (d, p1[1]), (d, p2[1]), p2])
    points = bezier(ts)
    bvx = [i[0] for i in points]
    bvy = [i[1] for i in points]
    return bvx, bvy


def make_bezier(xys):
    # This function originated from the plot.ly's documentation for python API.
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n - 1)

    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1 - t)**i for i in range(n)])
            coefs = [c * a * b for c, a,
                     b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef * p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier


def pascal_row(n):
    # This function originated from the plot.ly's documentation for python API.
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n // 2 + 1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result
