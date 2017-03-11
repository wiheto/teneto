
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import teneto
import scipy.stats as sps

def derive_with_weighted_pearson(data,method,postpro,params={},dimord='node,time',analysis_id=''):

    """

    Derives connectivity from the data. A lot of data is inherently built with edges (e.g. communication between two individuals).
    However other networks are derived from the covariance of time series (e.g. brain networks between two regions).

    Covariance based metrics deriving time-resolved networks can be done in multiple ways. There are other methods apart from covariance based.

    Derive a weight vector for each time point and then the corrrelation coefficient for each time point.

    A report is saved in ./report/[analysis_id]/derivation_report.html 

    :PARAMETERS:

    :data: input data. Default nodes=rows, time=columns. Change dimord if you want it the other way.
    :method: "distance","slidingwindow", "taperedslidingwindow". Alternatively, method can be a weight matrix of size time x time.
    :postpro: "no" (default). Other alternatives are: "all","fisher+boxcox+ztransform", "fischer","fisher+boxcox","boxcox", and more. See postpro_pipeline for more information.
    :params: dictionary of parameters for each method (see below).
    :data_dimord: can be 'node,time' or 'time,node'. People like to represent their data differently and this is an easy way to be sure that you are inputing the data in the correct way.
    :analysis_id: add to identify specfic analysis. Generated report will be placed in './report/' + analysis_id + '/derivation_report.html

    *When method = "distance"*

        Distance metric calculates 1/Distance metric weights, and scales between 0 and 1. W[t,t] is excluded from the scaling and then set to 1.

        params['distance'] = 'euclidean', 'hamming'. See teneto.utils.getDistanceFunction

    *When method = "cluster"* (not yet implemented)

        :params['cluster']: = 'kmean' (default). More will probabaly be added be made
        :params['k']: = integer or range of intergers. Needed when cluster='kmean'

    *When method = "taperedslidingwindow"*

        :params['windowsize']: = integer. Size of window.
        :params['distribution']: = Scipy distribution (e.g. 'norm','expon'). Any distribution here: https://docs.scipy.org/doc/scipy/reference/stats.html
        :params['distribution_params']: = list of each parameter, excluding the data "x" (in their scipy function order) to generate pdf.

        IMPORTANT: The data x should be considered to be centered at 0 and have a length of window size. (i.e. a window size of 5 entails x is [-2, -1, 0, 1, 2] a window size of 6 entails [-2.5, -1.5, 0.5, 0.5, 1.5, 2.5])
        Given x params['distribution_params'] contains the remaining parameters.

        e.g. normal distribution requires pdf(x, loc, scale) where loc=mean and scale=std. This means that the mean and std have to be provided in distribution_params.

        Say we have a gaussian distribution, a window size of 21 and params['distribution_params'] is [0,5].. This will lead to a gaussian with its peak at in the middle of each window with a standard deviation of 5.

        Instead, if we set params['distribution_params'] is [10,5] this will lead to a half gaussian with its peak at the final time point with a standard deviation of 5.

    *When method = "slidingwindow"*

        :params['windowsize']: = integer. Size of window.

    :RETURNS:

    Graphlet,DeriveInfo
        :Graphlet: representation (nodes x nodes x time)
        :DeriveInfo: dictionary containing information about derivation.

    :READ MORE:

    For more information about the weighted pearson method and how

    :SEE ALSO:

    *postpro_pipeline*, *gen_report*

    :HISTORY:

    Created - March 2017 - wht


    """
    report={}

    if dimord == 'node,time':
        data=data.transpose()

    if isinstance(method,str):
        if method == 'jackknife':
            w,report = weightfun_jackknife(data.shape[0],report)

        elif method == 'sliding window' or method == 'slidingwindow':
            w,report = weightfun_sliding_window(data.shape[0],params,report)

        elif method == 'tapered sliding window' or method == 'taperedslidingwindow':
            w,report = weightfun_tapered_sliding_window(data.shape[0],params,report)

        elif method == 'distance' or method == "spatial distance" or method == "node distance" or method == "nodedistance" or method == "spatialdistance":
            w,report = weightfun_spatial_distance(data,params,report)
        else:
            raise ValueError('Unrecognoized method. See derive_with_weighted_pearson documentation for predefined methods or enter own weight matrix')
    else:
        try:
            w=np.array(method)
        except:
            raise ValueError('Unrecognoized method. See derive_with_weighted_pearson documentation for predefined methods or enter own weight matrix')
        if w.shape[0] != w.shape[1]:
            raise ValueError("weight matrix should be square")
        if w.shape[0] != data.shape[0]:
            raise ValueError("weight matrix must equal number of time points")


    # Loop over each weight vector and calculate pearson correlation.
    # Note, should see if this can be made quicker in future.
    R = np.array([DescrStatsW(data,w[i,:]).corrcoef for i in range(0,w.shape[0])])
    # Make node,node,time
    R = R.transpose([1, 2, 0])

    # Correct jackknife direction
    if method == 'jackknife':
        R = R*-1
        R = R*-1

    if postpro != 'no':
        R,report=teneto.derive.postpro_pipeline(R,postpro,report)
        R[np.isinf(R)]=0

    teneto.derive.gen_report(report,'./report/' + analysis_id)
    return R

#def weightfun_upcoming(T,report):

#    w=np.ones([T,T])
#    np.fill_diagonal(w,0)
#    return w, report

def weightfun_sliding_window(T,params,report):

    w0=np.zeros(T)
    w0[0:params['windowsize']] = np.ones(params['windowsize'])
    w = np.array([np.roll(w0,i) for i in range(0,T+1-params['windowsize'])])
    report['method'] = 'slidingwindow'
    report['slidingwindow'] = params
    report['slidingwindow']['taper'] = 'untapered/uniform'
    return w, report

def weightfun_tapered_sliding_window(T,params,report):


    x=np.arange(-(params['windowsize']-1)/2,(params['windowsize'])/2)
    distribution_parameters = ','.join(map(str,params['distribution_params']))
    taper = eval('sps.' + params['distribution'] + '.pdf(x,' + distribution_parameters + ')')

    w0=np.zeros(T)
    w0[0:params['windowsize']] = taper
    w = np.array([np.roll(w0,i) for i in range(0,T+1-params['windowsize'])])
    report['method'] = 'slidingwindow'
    report['slidingwindow'] = params
    report['slidingwindow']['taper'] = taper
    report['slidingwindow']['taper_window'] = x
    return w, report

def weightfun_spatial_distance(data,params,report):
    distance = teneto.utils.getDistanceFunction(params['distance'])
    w=np.array([distance(data[n,:],data[t,:]) for n in np.arange(0,data.shape[0])  for t in np.arange(0,data.shape[0])])
    w=np.reshape(w,[data.shape[0],data.shape[0]])
    np.fill_diagonal(w,np.nan)
    if params['equation'] == '1/D':
        w=1/w
    w=(w-np.nanmin(w))/(np.nanmax(w)-np.nanmin(w))
    np.fill_diagonal(w,1)
    return w, report
