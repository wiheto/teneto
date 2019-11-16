"""Function to draw a circle_plot"""
import numpy as np
import math
from .slice_plot import make_bezier
from ..utils import check_input, graphlet2contact
import matplotlib.cm as cm


def circle_plot(netIn, ax, nodelabels=None, linestyle='k-', nodesize=1000, cmap='Set2'):
    r"""

    Function draws "circle plot" and exports axis handles

    Parameters
    -------------
    netIn : temporal network input (graphlet or contact)
    ax : matplotlib ax handles.
    nodelabels : list
        nodes labels. List of strings
    linestyle : str
        line style
    nodesize : int
        size of nodes
    cmap : str
        matplotlib colormap


    Returns
    -------
    ax : axis handle

    Example
    -------
    >>> import teneto
    >>> import numpy
    >>> import matplotlib.pyplot as plt
    >>> G = np.zeros([6, 6])
    >>> i = [0, 0, 0, 1, 2, 3, 4]
    >>> j = [3, 4, 5, 5, 4, 5, 5]
    >>> G[i, j] = 1
    >>> fig, ax = plt.subplots(1)
    >>> ax = teneto.plot.circle_plot(G, ax)
    >>> fig.show()

    .. plot::

        import teneto
        import numpy
        import matplotlib.pyplot as plt
        G = np.zeros([6, 6])
        i = [0, 0, 0, 1, 2, 3, 4]
        j = [3, 4, 5, 5, 4, 5, 5]
        G[i, j] = 1
        fig, ax = plt.subplots(1)
        teneto.plot.circle_plot(G, ax)
        fig.show()

    """
    # Get input type (C or G)
    inputType = check_input(netIn, conMat=1)
    if nodelabels is None:
        nodelabels = []
    # Convert C representation to G
    if inputType == 'M':
        shape = np.shape(netIn)
        edg = np.where(np.abs(netIn) > 0)
        contacts = [tuple([edg[0][i], edg[1][i]])
                    for i in range(0, len(edg[0]))]
        netIn = {}
        netIn['contacts'] = contacts
        netIn['netshape'] = shape
    elif inputType == 'G':
        netIn = graphlet2contact(netIn)
        inputType = 'C'

    if inputType == 'C':
        edgeList = [tuple(np.array(e[0:2]) + e[2] * netIn['netshape'][0])
                    for e in netIn['contacts']]
    elif inputType == 'M':
        edgeList = netIn['contacts']

    n = netIn['netshape'][0]
    # Get positions of node on unit circle
    posx = [math.cos((2 * math.pi * i) / n) for i in range(0, n)]
    posy = [math.sin((2 * math.pi * i) / n) for i in range(0, n)]
    # Get Bezier lines in a circle
    cmap = cm.get_cmap(cmap)(np.linspace(0, 1, n))
    for edge in edgeList:
        bvx, bvy = bezier_circle(
            (posx[edge[0]], posy[edge[0]]), (posx[edge[1]], posy[edge[1]]), 20)
        ax.plot(bvx, bvy, linestyle, zorder=0)
    for i in range(n):
        ax.scatter(posx[i], posy[i], s=nodesize, c=cmap[i], zorder=1)
    # Remove things that make plot unpretty
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)
    # make plot a square
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return ax


# Adapaption of bezier_points but for going in towards the centre of circle
def bezier_circle(p1, p2, pointN):
    ts = [t / pointN for t in range(pointN + 1)]
#    d=p1[0]-(max(p1[1],p2[1])-min(p1[1],p2[1]))/negxLim
    bezier = make_bezier([p1, (0, 0), p2])
    points = bezier(ts)
    bvx = [i[0] for i in points]
    bvy = [i[1] for i in points]
    return bvx, bvy
