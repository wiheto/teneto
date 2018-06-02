
import teneto
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


def test_sliceplot():
    G = teneto.generatenetwork.rand_binomial([4, 2], 0.5, 'graphlet', 'wu')
    fig, ax = plt.subplots(1)
    ax = teneto.plot.slice_plot(G, ax)
    plt.close(fig)


def test_circleplot():
    G = teneto.generatenetwork.rand_binomial([4, 2], 0.5, 'graphlet', 'wd')
    fig, ax = plt.subplots(1)
    ax = teneto.plot.circle_plot(G.mean(axis=-1), ax)
    plt.close(fig)


def test_stackplot():
    G = teneto.generatenetwork.rand_binomial([4, 2], 0.5, 'contact', 'wd')
    fig, ax = plt.subplots(1)
    ax = teneto.plot.graphlet_stack_plot(G, ax, q=1)
    plt.close(fig)
