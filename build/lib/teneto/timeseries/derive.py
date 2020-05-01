"""derive: different methods to derive time-varying functional connectivity"""


import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from ..utils import set_diagonal, get_distance_function
from .postprocess import postpro_pipeline
from .report import gen_report
import scipy.stats as sps
from scipy.signal import hilbert


def derive_temporalnetwork(data, params):
    """
    Derives connectivity from the data.

    A lot of data is inherently built with edges
     (e.g. communication between two individuals).
    However other networks are derived from the covariance of time series
     (e.g. brain networks between two regions).

    Covariance based metrics deriving time-resolved networks can be done in multiple ways.
     There are other methods apart from covariance based.

    Derive a weight vector for each time point and then the corrrelation coefficient
     for each time point.

    Paramters
    ----------

    data : array
        Time series data to perform connectivity derivation on.
        (Default dimensions are: (time as rows, nodes as columns).
        Change params{'dimord'} if you want it the other way (see below).

    params : dict
        Parameters for each method (see below).

    Necessary paramters
    ===================

    method : str
        method: "distance","slidingwindow", "taperedslidingwindow",
     "jackknife", "multiplytemporalderivative".
     Alternatively, method can be a weight matrix of size time x time.

    **Different methods have method specific paramaters (see below)**

    Params for all methods (optional)
    =================================

    postpro : "no" (default).
    Other alternatives are: "fisher", "boxcox", "standardize"
     and any combination seperated by a + (e,g, "fisher+boxcox").
      See postpro_pipeline for more information.
    dimord : str
        Dimension order: 'node,time' (default) or 'time,node'.
        People like to represent their data differently and this is an easy way
        to be sure that you are inputing the data in the correct way.
    analysis_id : str or int
        add to identify specfic analysis.
        Generated report will be placed in './report/' + analysis_id + '/derivation_report.html
    report : bool
        False by default.
        If true, A report is saved in ./report/[analysis_id]/derivation_report.html if "yes"
    report_path : str
        String where the report is saved.
        Default is ./report/[analysis_id]/derivation_report.html

    Methods specific parameters
    ===========================

    method == "distance"
    ~~~~~~~~~~~~~~~~~~~

    Distance metric calculates 1/Distance metric weights, and scales between 0 and 1.
    W[t,t] is excluded from the scaling and then set to 1.

    params['distance']: str
        Distance metric (e.g. 'euclidean'). See teneto.utils.get_distance_function for more info

    When method == "slidingwindow"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    params['windowsize'] : int
        Size of window.

    When method == "taperedslidingwindow"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    params['windowsize'] : int
        Size of window.
    params['distribution'] : str
        Scipy distribution (e.g. 'norm','expon'). Any distribution here: https://docs.scipy.org/doc/scipy/reference/stats.html
    params['distribution_params'] : dict
        Dictionary of distribution parameter, excluding the data "x" to generate pdf.

        The data x should be considered to be centered at 0 and have a length of window size.
         (i.e. a window size of 5 entails x is [-2, -1, 0, 1, 2] a window size of 6 entails [-2.5, -1.5, 0.5, 0.5, 1.5, 2.5])
        Given x params['distribution_params'] contains the remaining parameters.

        e.g. normal distribution requires pdf(x, loc, scale) where loc=mean and scale=std.

        Say we have a gaussian distribution, a window size of 21 and params['distribution_params'] = {'loc': 0, 'scale': 5}.
         This will lead to a gaussian with its peak at in the middle of each window with a standard deviation of 5.

    When method == "temporalderivative"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    params['windowsize'] : int
        Size of window.

    When method == "jackknife"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    No parameters are necessary.

    Optional parameters:

    params['weight-var'] : array, (optional)
        NxN array to weight the JC estimates (standerdized-JC*W). If weightby is selected, do not standerdize in postpro.
    params['weight-mean'] : array, (optional)
        NxN array to weight the JC estimates (standerdized-JC+W). If weightby is selected, do not standerdize in postpro.


    When method == 'instantaneousphasesync'
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    No parameters are necessary.

    Returns
    -------

    G : array
        Connectivity estimates (nodes x nodes x time)


    READ MORE
    ---------
    About the general weighted pearson approach used for most methods, see:
    Thompson & Fransson (2019) A common framework for the problem of deriving estimates of dynamic functional brain connectivity.
    Neuroimage. (https://doi.org/10.1016/j.neuroimage.2017.12.057)

    SEE ALSO
    --------

    *postpro_pipeline*, *gen_report*

    """
    report = {}

    if 'dimord' not in params.keys():
        params['dimord'] = 'node,time'

    if 'report' not in params.keys():
        params['report'] = False

    if 'analysis_id' not in params.keys():
        params['analysis_id'] = ''

    if 'postpro' not in params.keys():
        params['postpro'] = 'no'

    if params['report'] == 'yes' or params['report']:

        if 'analysis_id' not in params.keys():
            params['analysis_id'] = ''

        if 'report_path' not in params.keys():
            params['report_path'] = './report/' + params['analysis_id']

        if 'report_filename' not in params.keys():
            params['report_filename'] = 'derivation_report.html'

    if params['dimord'] == 'node,time':
        data = data.transpose()

    sw_alternatives = ['sliding window', 'slidingwindow']
    tsw_alternatives = ['tapered sliding window', 'taperedslidingwindow']
    sd_alternatives = ['distance', "spatial distance",
                       "node distance", "nodedistance", "spatialdistance"]
    mtd_alternatives = ['mtd', 'multiply temporal derivative',
                        'multiplytemporalderivative', 'temporal derivative', "temporalderivative"]
    ip_alternatives = ['instantaneousphasesync', 'ips']
    jc_alternatives = ['jackknife', 'jackknifecorrelation', 'jc']
    if isinstance(params['method'], str):
        if params['method'] in jc_alternatives:
            weights, report = _weightfun_jackknife(data.shape[0], report)
            relation = 'weight'
        elif params['method'] in sw_alternatives:
            weights, report = _weightfun_sliding_window(
                data.shape[0], params, report)
            relation = 'weight'
        elif params['method'] in tsw_alternatives:
            weights, report = _weightfun_tapered_sliding_window(
                data.shape[0], params, report)
            relation = 'weight'
        elif params['method'] in sd_alternatives:
            weights, report = _weightfun_spatial_distance(data, params, report)
            relation = 'weight'
        elif params['method'] in mtd_alternatives:
            R, report = _temporal_derivative(data, params, report)
            relation = 'coupling'
        elif params['method'] in ip_alternatives:
            R, report = _instantaneous_phasesync(data, params, report)
            relation = 'coupling'
        else:
            raise ValueError(
                'Unrecognoized method. See derive_with_weighted_pearson documentation for predefined methods or enter own weight matrix')
    else:
        try:
            weights = np.array(params['method'])
            relation = 'weight'
        except:
            raise ValueError(
                'Unrecognoized method. See documentation for predefined methods')
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("weight matrix should be square")
        if weights.shape[0] != data.shape[0]:
            raise ValueError("weight matrix must equal number of time points")

    if relation == 'weight':
        # Loop over each weight vector and calculate pearson correlation.
        # Note, should see if this can be made quicker in future.
        R = np.array(
            [DescrStatsW(data, weights[i, :]).corrcoef for i in range(0, weights.shape[0])])
        # Make node,node,time
        R = R.transpose([1, 2, 0])

    # Correct jackknife direction
    if params['method'] == 'jackknife':
        # Correct inversion
        R = R * -1
        jc_z = 0
        if 'weight-var' in params.keys():
            R = np.transpose(R, [2, 0, 1])
            R = (R - R.mean(axis=0)) / R.std(axis=0)
            jc_z = 1
            R = R * params['weight-var']
            R = R.transpose([1, 2, 0])
        if 'weight-mean' in params.keys():
            R = np.transpose(R, [2, 0, 1])
            if jc_z == 0:
                R = (R - R.mean(axis=0)) / R.std(axis=0)
            R = R + params['weight-mean']
            R = np.transpose(R, [1, 2, 0])
        R = set_diagonal(R, 1)

    if params['postpro'] != 'no':
        R, report = postpro_pipeline(
            R, params['postpro'], report)
        R = set_diagonal(R, 1)

    if params['report'] == 'yes' or params['report']:
        gen_report(report, params['report_path'], params['report_filename'])
    return R


def _weightfun_jackknife(T, report):
    """Creates the weights for the jackknife method. See func: teneto.timeseries.derive_temporalnetwork."""

    weights = np.ones([T, T])
    np.fill_diagonal(weights, 0)
    report['method'] = 'jackknife'
    report['jackknife'] = ''
    return weights, report


def _weightfun_sliding_window(T, params, report):
    """Creates the weights for the sliding window method. See func: teneto.timeseries.derive_temporalnetwork."""
    weightat0 = np.zeros(T)
    weightat0[0:params['windowsize']] = np.ones(params['windowsize'])
    weights = np.array([np.roll(weightat0, i)
                        for i in range(0, T + 1 - params['windowsize'])])
    report['method'] = 'slidingwindow'
    report['slidingwindow'] = params
    report['slidingwindow']['taper'] = 'untapered/uniform'
    return weights, report


def _weightfun_tapered_sliding_window(T, params, report):
    """Creates the weights for the tapered method. See func: teneto.timeseries.derive_temporalnetwork."""
    x = np.arange(-(params['windowsize'] - 1) / 2, (params['windowsize']) / 2)
    taper = getattr(sps, params['distribution']).pdf(
        x, **params['distribution_params'])

    weightat0 = np.zeros(T)
    weightat0[0:params['windowsize']] = taper
    weights = np.array([np.roll(weightat0, i)
                        for i in range(0, T + 1 - params['windowsize'])])
    report['method'] = 'slidingwindow'
    report['slidingwindow'] = params
    report['slidingwindow']['taper'] = taper
    report['slidingwindow']['taper_window'] = x
    return weights, report


def _weightfun_spatial_distance(data, params, report):
    """Creates the weights for the spatial distance method. See func: teneto.timeseries.derive_temporalnetwork."""
    distance = get_distance_function(params['distance'])
    weights = np.array([distance(data[n, :], data[t, :]) for n in np.arange(
        0, data.shape[0]) for t in np.arange(0, data.shape[0])])
    weights = np.reshape(weights, [data.shape[0], data.shape[0]])
    np.fill_diagonal(weights, np.nan)
    weights = 1 / weights
    weights = (weights - np.nanmin(weights)) / \
        (np.nanmax(weights) - np.nanmin(weights))
    np.fill_diagonal(weights, 1)
    return weights, report


def _temporal_derivative(data, params, report):
    """Performs mtd method. See func: teneto.timeseries.derive_temporalnetwork."""
    # Data should be timexnode
    report = {}

    # Derivative
    tdat = data[1:, :] - data[:-1, :]
    # Normalize
    tdat = tdat / np.std(tdat, axis=0)
    # Coupling
    coupling = np.array([tdat[:, i] * tdat[:, j] for i in np.arange(0,
                                                                    tdat.shape[1]) for j in np.arange(0, tdat.shape[1])])
    coupling = np.reshape(
        coupling, [tdat.shape[1], tdat.shape[1], tdat.shape[0]])
    # Average over window using strides
    shape = coupling.shape[:-1] + (coupling.shape[-1] -
                                   params['windowsize'] + 1, params['windowsize'])
    strides = coupling.strides + (coupling.strides[-1],)
    coupling_windowed = np.mean(np.lib.stride_tricks.as_strided(
        coupling, shape=shape, strides=strides), -1)

    report = {}
    report['method'] = 'temporalderivative'
    report['temporalderivative'] = {}
    report['temporalderivative']['windowsize'] = params['windowsize']

    return coupling_windowed, report


def _instantaneous_phasesync(data, params, report):
    """Derivce instantaneous phase synchrony. See func: teneto.timeseries.derive_temporalnetwork."""
    analytic_signal = hilbert(data.transpose())
    instantaneous_phase = np.angle(analytic_signal)
    ips = np.zeros([data.shape[1], data.shape[1], data.shape[0]])
    for n in range(data.shape[1]):
        for m in range(data.shape[1]):
            ips[n, m, :] = 1 - np.sin(np.abs(instantaneous_phase[n] - instantaneous_phase[m])/2)

    report = {}
    report['method'] = 'instantaneousphasesync'
    report['instantaneousphasesync'] = {}

    return ips, report
