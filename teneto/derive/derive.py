
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import teneto
import scipy.stats as sps

def derive(data,params):

    """

    Derives connectivity from the data. A lot of data is inherently built with edges (e.g. communication between two individuals).
    However other networks are derived from the covariance of time series (e.g. brain networks between two regions).

    Covariance based metrics deriving time-resolved networks can be done in multiple ways. There are other methods apart from covariance based.

    Derive a weight vector for each time point and then the corrrelation coefficient for each time point.

    :PARAMETERS:

    :data: input data. (Default: time times are rows, nodes are columns). Change params{'dimord'} if you want it the other way (see below).
    :params: dictionary of parameters for each method (see below).

    *params for all methods (necessary)*

    :method: "distance","slidingwindow", "taperedslidingwindow", "jackknife", "temporalderivative". Alternatively, method can be a weight matrix of size time x time.

    *params for all methods (optional)*

    :postpro: "no" (default). Other alternatives are: "fisher", "boxcox", "standardize" and any combination seperated by a + (e,g, "fisher+boxcox"). See postpro_pipeline for more information.
    :data_dimord: 'node,time' (default) or 'time,node'. People like to represent their data differently and this is an easy way to be sure that you are inputing the data in the correct way.
    :analysis_id: add to identify specfic analysis. Generated report will be placed in './report/' + analysis_id + '/derivation_report.html
    :report: "yes" (default) or "no". A report is saved in ./report/[analysis_id]/derivation_report.html if "yes"


    *When method == "distance"*

        Distance metric calculates 1/Distance metric weights, and scales between 0 and 1. W[t,t] is excluded from the scaling and then set to 1.

        params['distance'] = 'euclidean', 'hamming'. See teneto.utils.getDistanceFunction

    *When method == "slidingwindow"*

        :params['windowsize']: = integer. Size of window.

    *When method == "taperedslidingwindow"*

        :params['windowsize']: = integer. Size of window.
        :params['distribution']: = Scipy distribution (e.g. 'norm','expon'). Any distribution here: https://docs.scipy.org/doc/scipy/reference/stats.html
        :params['distribution_params']: = list of each parameter, excluding the data "x" (in their scipy function order) to generate pdf.

        IMPORTANT: The data x should be considered to be centered at 0 and have a length of window size. (i.e. a window size of 5 entails x is [-2, -1, 0, 1, 2] a window size of 6 entails [-2.5, -1.5, 0.5, 0.5, 1.5, 2.5])
        Given x params['distribution_params'] contains the remaining parameters.

        e.g. normal distribution requires pdf(x, loc, scale) where loc=mean and scale=std. This means that the mean and std have to be provided in distribution_params.

        Say we have a gaussian distribution, a window size of 21 and params['distribution_params'] is [0,5].. This will lead to a gaussian with its peak at in the middle of each window with a standard deviation of 5.

        Instead, if we set params['distribution_params'] is [10,5] this will lead to a half gaussian with its peak at the final time point with a standard deviation of 5.

    *When method == "temporalderivative"*

        :params['windowsize']: = integer. Size of window.

    *When method == "jackknife"*

        No extra parameters needed.


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

    if 'dimord' not in params.keys():
        params['dimord']='time,node'

    if 'report' not in params.keys():
        params['report']='yes'

    if 'analysis_id' not in params.keys():
        params['analysis_id']=''

    if 'postpro' not in params.keys():
        params['postpro']='no'

    if params['dimord'] == 'node,time':
        data=data.transpose()

    if isinstance(params['method'],str):
        if params['method'] == 'jackknife':
            w,report = weightfun_jackknife(data.shape[0],report)
            relation = 'weight'
        elif params['method'] == 'sliding window' or params['method'] == 'slidingwindow':
            w,report = weightfun_sliding_window(data.shape[0],params,report)
            relation = 'weight'
        elif params['method'] == 'tapered sliding window' or params['method'] == 'taperedslidingwindow':
            w,report = weightfun_tapered_sliding_window(data.shape[0],params,report)
            relation = 'weight'
        elif params['method'] == 'distance' or params['method'] == "spatial distance" or params['method'] == "node distance" or params['method'] == "nodedistance" or params['method'] == "spatialdistance":
            w,report = weightfun_spatial_distance(data,params,report)
            relation = 'weight'
        elif params['method'] == 'temporal derivative' or params['method'] == "temporalderivative":
            R,report = temporal_derivative(data,params,report)
            relation = 'coupling'
        else:
            raise ValueError('Unrecognoized method. See derive_with_weighted_pearson documentation for predefined methods or enter own weight matrix')
    else:
        try:
            w=np.array(method)
            relation = 'weight'
        except:
            raise ValueError('Unrecognoized method. See derive_with_weighted_pearson documentation for predefined methods or enter own weight matrix')
        if w.shape[0] != w.shape[1]:
            raise ValueError("weight matrix should be square")
        if w.shape[0] != data.shape[0]:
            raise ValueError("weight matrix must equal number of time points")


    if relation == 'weight':
        # Loop over each weight vector and calculate pearson correlation.
        # Note, should see if this can be made quicker in future.
        R = np.array([DescrStatsW(data,w[i,:]).corrcoef for i in range(0,w.shape[0])])
        # Make node,node,time
        R = R.transpose([1, 2, 0])

    # Correct jackknife direction
    if params['method'] == 'jackknife':
        R = R*-1

    if params['postpro'] != 'no':
        R,report=teneto.derive.postpro_pipeline(R,params['postpro'],report)
        R[np.isinf(R)]=0

    if params['report']=='yes':
        teneto.derive.gen_report(report,'./report/' + params['analysis_id'])
    return R

def weightfun_jackknife(T,report):

    w=np.ones([T,T])
    np.fill_diagonal(w,0)
    return w, report

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
    w=1/w
    w=(w-np.nanmin(w))/(np.nanmax(w)-np.nanmin(w))
    np.fill_diagonal(w,1)
    return w, report



def temporal_derivative(data,params,report):

    #Data should be timexnode
    report = {}

    #Derivative
    tdat = data[1:,:]-data[:-1,:]
    #Normalize
    tdat = tdat/np.std(tdat,axis=0)
    #Coupling
    coupling = np.array([tdat[:,i]*tdat[:,j] for i in np.arange(0,tdat.shape[1]) for j in np.arange(0,tdat.shape[1])])
    coupling=np.reshape(coupling,[tdat.shape[1],tdat.shape[1],tdat.shape[0]])
    #Average over window using strides
    shape = coupling.shape[:-1] + (coupling.shape[-1] - params['windowsize'] + 1, params['windowsize'])
    strides = coupling.strides + (coupling.strides[-1],)
    coupling_windowed = np.mean(np.lib.stride_tricks.as_strided(coupling, shape=shape, strides=strides),-1)

    report = {}
    report['method'] = 'temporalderivative'
    report['temporalderivative']={}
    report['temporalderivative']['windowsize'] = params['windowsize']

    return coupling_windowed,report
