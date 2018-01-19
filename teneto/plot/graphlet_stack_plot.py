import matplotlib.pyplot as plt
import numpy as np
from teneto.utils import contact2graphlet, checkInput

from scipy import ndimage

plt.rcParams['axes.facecolor'] = 'white'


def graphlet_stack_plot(netIn, ax, q=10, cmap='Reds', gridcolor='k', borderwidth=2, bordercolor=[0, 0, 0], Fs=1, timeunit='', t0=1, sharpen='yes', vminmax='minmax'):
    '''
    Returns matplotlib axis handle for graphlet_stack_plot. This is a row of transformed connectivity matrices to look like a 3D stack.
    **PARAMETERS**
    :netIn: network input (graphlet or contact)
    :ax: matplotlib ax handles.
    :q: quality. Increaseing this will lead to smoother axis but take up more memory.
    :cmap: colormap (matplotlib) of graphlets
    :Fs: sampling rate. Same as contact-representation (if netIn is contact, and input is unset, contact dictionary is used)
    :timeunit: for plotting. Same as contact-representation (if netIn is contact, and input is unset, contact dictionary is used)
    :t0: what should the first time point be called. Should be integer. Default 1.
    :gridcolor: The color of the grid section of the graphlets. Set to 'none' if not wanted.
    :borderwidth: Integer that scales the size of border. (at the moment it cannot be set to 0.)
    :bordorcolor: color of the border (at the moment it must be in RGB values between 0 and 1 -> this will be changed sometime in the future)
    :vminmax: 'maxabs', 'minmax' (default), or list/array with length of 2. Specifies the min and max colormap value of graphlets. Maxabs entails [-max(abs(G)),max(abs(G))], minmax entails [min(G), max(G)].
    **OUTPUT**
    :ax: matplotlib ax handle
    **NOTE**
    This function can require a lot of RAM with larger networks.
    At the momenet bordercolor cannot be set to zero. To remove border, set bordorwidth=1 and bordercolor=[1,1,1] for temporay workaround.
    **SEE ALSO**
    - *circle_plot*
    - *slice_plot*
    **HISTORY**
    :Created: Dec 2016, WHT
    '''

    # Get input type (C, G, TO)
    inputType = checkInput(netIn)

    # Convert TO to C representation
    if inputType == 'TO':
        netIn = netIn.contact
        inputType = 'C'
    # Convert C representation to G
    if inputType == 'C':
        nettype = netIn['nettype']
        if timeunit == '':
            timeunit = netIn['timeunit']
        if t0 == 1:
            t0 = netIn['t0']
        if Fs == 1:
            Fs = netIn['Fs']
        netIn = contact2graphlet(netIn)

    if timeunit != '':
        timeunit = ' (' + timeunit + ')'

    if not isinstance(borderwidth, int):
        borderwidth = int(borderwidth)
        print('Warning: borderwidth should be an integer. Converting to integer.')

    # x and y ranges for each of the graphlet plots
    v = np.arange(0, netIn.shape[0] + 1)
    vr = np.arange(netIn.shape[0], -1, -1)
    # Preallocatie matrix

    if vminmax == '' or vminmax == 'absmax' or vminmax == 'maxabs':
        vminmax = [-np.nanmax(np.abs(netIn)), np.nanmax(np.abs(netIn))]
    elif vminmax == 'minmax':
        vminmax = [np.nanmin(netIn), np.nanmax(netIn)]

    qb = q * borderwidth
    figmat = np.zeros([80 * q + (qb * 2), int(((netIn.shape[-1]) *
                                               (80 * q) + (qb * 2)) - ((netIn.shape[-1] - 1) * q * 80) / 2), 4])
    for n in range(0, netIn.shape[-1]):
        # Create graphlet
        figtmp, axtmp = plt.subplots(1, facecolor='white', figsize=(q, q), dpi=80)
        axtmp.pcolormesh(v, vr, netIn[:, :, n], cmap=cmap, edgecolor=gridcolor,
                         linewidth=q * 2, vmin=vminmax[0], vmax=vminmax[1])
        axtmp.set_xticklabels('')
        axtmp.set_yticklabels('')
        axtmp.set_xticks([])
        axtmp.set_yticks([])
        x0, x1 = axtmp.get_xlim()
        y0, y1 = axtmp.get_ylim()
        axtmp.set_aspect((x1 - x0) / (y1 - y0))
        axtmp.spines['left'].set_visible(False)
        axtmp.spines['right'].set_visible(False)
        axtmp.spines['top'].set_visible(False)
        axtmp.spines['bottom'].set_visible(False)
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)

        # Convert graphlet to RGB values
        figtmp.canvas.draw()
        figmattmp = np.fromstring(
            figtmp.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        figmattmp = figmattmp.reshape(
            figtmp.canvas.get_width_height()[::-1] + (3,))

        # Close figure for memory
        plt.close(figtmp)

        # Manually add a border

        figmattmp_withborder = np.zeros(
            [figmattmp.shape[0] + (qb * 2), figmattmp.shape[1] + (qb * 2), 3]) + (np.array(bordercolor) * 255)
        figmattmp_withborder[qb:-qb, qb:-qb, :] = figmattmp

        # Make corners rounded. First make a circle and then take the relevant quarter for each corner.
        y, x = np.ogrid[-qb: qb + 1, -qb: qb + 1]
        mask = x * x + y * y <= qb * qb
        # A little clumsy. Should improve
        Mq1 = np.vstack([[mask[:qb, :qb] == 0], [mask[:qb, :qb] == 0], [
                        mask[:qb, :qb] == 0]]).transpose([1, 2, 0])
        figmattmp_withborder[:qb, :qb, :][Mq1] = 255
        Mq1 = np.vstack([[mask[:qb, -qb:] == 0], [mask[:qb, -qb:]
                                                  == 0], [mask[:qb, -qb:] == 0]]).transpose([1, 2, 0])
        figmattmp_withborder[:qb, -qb:, :][Mq1] = 255
        Mq1 = np.vstack([[mask[-qb:, :qb] == 0], [mask[-qb:, :qb]
                                                  == 0], [mask[-qb:, :qb] == 0]]).transpose([1, 2, 0])
        figmattmp_withborder[-qb:, :qb, :][Mq1] = 255
        Mq1 = np.vstack([[mask[-qb:, -qb:] == 0], [mask[-qb:, -qb:]
                                                   == 0], [mask[-qb:, -qb:] == 0]]).transpose([1, 2, 0])
        figmattmp_withborder[-qb:, -qb:, :][Mq1] = 255

        #scale and sheer
        scale = np.matrix([[1.5, 0, 0], [0, 3, 0], [0, 0, 1]])
        sheer = np.matrix([[1, np.tan(np.pi / 12), 0], [0, 1, 0], [0, 0, 1]])

        # apply affine transformation
        figmattmp = ndimage.affine_transform(
            figmattmp_withborder, sheer * (scale), offset=[-35 * q, 0, 0], cval=255)

        # At the moment the alpha part does not work if the background colour is anything but white.
        # Also used for detecting where the graphlets are in the image.
        trans = np.where(np.sum(figmattmp, axis=2) == 255 * 3)
        alphamat = np.ones([figmattmp.shape[0], figmattmp.shape[0]])
        alphamat[trans[0], trans[1]] = 0
        figmattmp = np.dstack([figmattmp, alphamat])

        # Add graphlet to matrix
        if n == 0:
            figmat[:, n * (80 * q):((n + 1) * (80 * q) + (qb * 2))] = figmattmp
        else:
            figmat[:, n * (80 * q) - int((n * q * 80) / 2):int(((n + 1)
                                                                * (80 * q) + (qb * 2)) - (n * q * 80) / 2)] = figmattmp

    # Fix colours - due to imshows weirdness when taking nxnx3
    figmat[:, :, 0:3] = figmat[:, :, 0:3] / 255
    # Cut end of matrix off that isn't need
    figmat = figmat[:, :-int((q / 2) * 80), :]
    fid = np.where(figmat[:, :, -1] > 0)
    fargmin = np.argmin(fid[0])
    ymax = np.max(fid[0])
    yright = np.max(np.where(figmat[:, fid[1][fargmin], -1] > 0))
    xtickloc = np.where(figmat[ymax, :, -1] > 0)[0]
    # In case there are multiple cases of xtickloc in same graphlet (i.e. they all have the same lowest value)
    xtickloc = np.delete(xtickloc, np.where(np.diff(xtickloc) == 1)[0] + 1)

    fid = np.where(figmat[:, :, -1] > 0)
    ymin = np.min(fid[0])
    topfig = np.where(figmat[ymin, :, -1] > 0)[0]
    topfig = topfig[0:len(topfig):int(len(topfig) / netIn.shape[-1])]

    # Make squares of non transparency around each figure (this fixes transparency issues when white is in the colormap)
    # for n in range(0,len(topfig)):
    # fid=np.where(figmat[ymin:ymax,xtickloc[n]:topfig[n],-1]==0)
    # figmat[ymin:ymax,xtickloc[n]:topfig[n],:3][fid[0],fid[1]]=1
    # figmat[ymin+q:ymax-q,xtickloc[n]+q:topfig[n]-q,-1]=1

    # Create figure
    # Sharped edges of figure with median filter
    if sharpen == 'yes':
        figmat[:, :, :-1] = ndimage.median_filter(figmat[:, :, :-1], 3)
    ax.imshow(figmat[:, :, :-1], zorder=1)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xticks([])
    ax.set_yticks([])

    L = int((((netIn.shape[-1] - 3) + 1) * (80 * q) +
             (qb * 2)) - ((netIn.shape[-1] - 3) * q * 80) / 2 - q)
    [ax.plot(range(topfig[i], xt), np.zeros(len(range(topfig[i], xt))) + yright,
             color='k', linestyle=':', zorder=2) for i, xt in enumerate(xtickloc[1:])]
    ax.plot(range(0, L), np.zeros(L) + ymax,
            color='k', linestyle=':', zorder=2)
    [ax.plot(np.zeros(q * 10) + xt, np.arange(ymax, ymax + q * 10),
             color='k', linestyle=':', zorder=2) for xt in xtickloc]
    [ax.text(xt, ymax + q * 20, str(round((i + t0) * Fs, 5)),
             horizontalalignment='center',) for i, xt in enumerate(xtickloc)]

    ylim = ax.axes.get_ylim()
    xlim = ax.axes.get_xlim()
    ax.set_ylim(ylim[0] + q * 15, 0)
    ax.set_xlim(xlim[0] - q * 20, xlim[1])
    ax.set_xlabel('Time' + timeunit)
    return ax
