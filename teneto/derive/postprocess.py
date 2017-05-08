import numpy as np
import scipy as sp
import teneto

def postpro_fisher(R,report={}):
    # Due to rounding errors
    R[R<-0.99999999999999]=-1
    R[R>0.99999999999999]=1
    R_z= 0.5*np.log((1+R)/(1-R))
    report['fisher']={}
    report['fisher']['performed'] = 'yes'
    #report['fisher']['diagonal'] = 'zeroed'
    return R_z,report

def postpro_boxcox(R,report={}):
    # Note the min value of all time series will now be at least 1. Making the magnitude based metrics hard.
    minR = 1 - np.nanmin(R)
    R = R + minR
    ind=np.triu_indices(R.shape[0],k=1)

    bc = np.array([sp.stats.boxcox(np.squeeze(R[ind[0][n],ind[1][n],:])) for n in range(0,len(ind[0]))])


    R_bc = np.zeros(R.shape)
    R_bc[ind[0],ind[1],:]=np.vstack(bc[:,0])
    R_bc[ind[1],ind[0],:]=np.vstack(bc[:,0])


    bccheck = np.array(np.transpose(R_bc,[2,0,1]))
    bccheck = (bccheck - bccheck.mean(axis=0)) / bccheck.std(axis=0)
    bccheck = np.squeeze(np.mean(bccheck,axis=0))
    np.fill_diagonal(bccheck,0)


    report['boxcox']={}
    report['boxcox']['performed'] = 'yes'
    report['boxcox']['lambda']=[tuple([ind[0][n],ind[1][n],bc[n,-1]]) for n in range(0,len(ind[0]))]
    report['boxcox']['shift']=minR
    report['boxcox']['shited_to']=1

    if np.sum(np.isnan(bccheck))>0:
        report['boxcox']={}
        report['boxcox']['performed'] = 'FAILED'
        report['boxcox']['failure_reason']='Box cox transform is returning edges with uniform values through time. This is probabaly due to one or more outliers or a very skewed distribution. Have you corrected for all possible sources of noise (e.g. movement)? If yes, then this time-series might not be able to make Gaussian without additional transformations beforehand.'
        report['boxcox']['failure_consequence']='Box cox transform was skipped from the postprocess pipeline.'
        R_bc = R - minR
        print("TENETO WARNING: Box Cox transform fauked to make normal distribution of the data. Probabaly due to outliers in the connectivity time series. Have all different artefacts been corrected for? See report for more details. \n Box Cox transform not performed.")



    return R_bc,report

def postpro_standardize(R,report={}):
    # First make trailing dimension nodal.
    R = np.transpose(R,[2,0,1])
    Z = (R - R.mean(axis=0)) / R.std(axis=0)
    Z = np.transpose(Z,[1,2,0])
    report['standardize']={}
    report['standardize']['performed'] = 'yes'
    report['standardize']['method'] = 'starndard score'
    return Z,report

def postpro_pipeline(R,pipeline,report=None):

    """

    :PARAMETERS:

    :R: pearson correlation value in temporal matrix form (node,node,time)
    :pipeline: list or string (if string, each steps seperated by + sign).

        :options: 'fischer','boxcox','standardize'

        Each of the above 3 can be specified. If fischer is used, it must be before boxcox. If standardize is used it must be after boxcox and fischer.

    :report: default is empty. If non-empty, appended to report.

    :OUTPUT:

    :dataOut: postprocessed data
    :postprocessing_info: dictionary of information about postprocessing (e.g lambda parameters for boxcox)

    """

    if report == None:
        report={}

    if isinstance(pipeline,str):
        pipeline=pipeline.split('+')

    report['postprocess'] = []
    for p in pipeline:
        report['postprocess'].append(p)
        R,report = eval('teneto.derive.postpro_' + p + '(R,report)')
    return R,report
