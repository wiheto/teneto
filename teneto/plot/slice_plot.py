# Main function to draw a slice_graph
import matplotlib.pyplot as plt
import numpy as np
from teneto.utils import *


def slice_plot(netIn, ax, nLabs='', tLabs='', timeunit='', linestyle='k-', cmap=None, nodesize=100):
    '''

    Fuction draws "slice graph" and exports axis handles


    **PARAMETERS**

    :netIn: temporal network input (graphlet or contact)
    :ax: matplotlib figure handles.
    :nLabs: nodes labels. List of strings.
    :tLabs: labels of dimension Graph is expressed across. List of strings.
    :timeunit: unit time axis is in.
    :linestyle: line style of Bezier curves.
    :nodesize: size of nodes


    **OUTPUT**

    :ax: axis handle of slice graph


    **SEE ALSO**

    - *circle_plot*
    - *graphlet_stack_plot*


    **HISTORY**

    :modified: Dec 2016, WHT (documentation, improvments)
    :created: Sept 2016, WHT

    '''
    # Get input type (C or G)
    inputType = checkInput(netIn)
    nettype = 'xx'
    # Convert C representation to G

    if inputType == 'G':
        cfg = {}
        netIn = graphlet2contact(netIn)
        inputType = 'C'
    edgeList = [tuple(np.array(e[0:2]) + e[2] * netIn['netshape'][0])
                for e in netIn['contacts']]

    if nLabs != '' and len(nLabs) == netIn['netshape'][0]:
        pass
    elif nLabs != '' and len(nLabs) != netIn['netshape'][0]:
        raise ValueError('specified node label length does not match netshape')
    elif nLabs == '' and netIn['nLabs'] == '':
        nLabs = np.arange(1, netIn['netshape'][0] + 1)
    else:
        nLabs = netIn['nLabs']

    if tLabs != '' and len(tLabs) == netIn['netshape'][-1]:
        pass
    elif tLabs != '' and len(tLabs) != netIn['netshape'][-1]:
        raise ValueError('specified time label length does not match netshape')
    elif tLabs == '' and str(netIn['t0']) == '':
        tLabs = np.arange(1, netIn['netshape'][-1] + 1)
    else:
        tLabs = np.arange(netIn['t0'], netIn['Fs'] *
                          netIn['netshape'][-1] + netIn['t0'], netIn['Fs'])

    if timeunit == '':
        timeunit = netIn['timeunit']

    timeNum = len(tLabs)
    nodeNum = len(nLabs)
    pos = []
    posy = np.tile(list(range(0, nodeNum)), timeNum)
    posx = np.repeat(list(range(0, timeNum)), nodeNum)

    node_plot_attr = {}
    if cmap:
        node_plot_attr['cmap'] = cmap


    # plt.plot(points)
    # Draw Bezier vectors around egde positions
    for edge in edgeList:
        bvx, bvy = bezier_points(
            (posx[edge[0]], posy[edge[0]]), (posx[edge[1]], posy[edge[1]]), nodeNum, 20)
        ax.plot(bvx, bvy, linestyle)
    ax.set_yticks(range(0, len(nLabs)))
    ax.set_xticks(range(0, len(tLabs)))
    ax.set_yticklabels(nLabs)
    ax.set_xticklabels(tLabs)
    ax.grid()
    ax.set_frame_on(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([min(posx) - 1, max(posx) + 1])
    ax.set_ylim([min(posy) - 1, max(posy) + 1])
    ax.scatter(posx, posy, s=nodesize, c=posy, zorder=10, **node_plot_attr)
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


# These two functions originated from the plot.ly's documentation for python API.
# They create points along a curve.
def make_bezier(xys):
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
