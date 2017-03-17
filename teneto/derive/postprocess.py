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

    report['boxcox']={}
    report['boxcox']['performed'] = 'yes'
    report['boxcox']['lambda']=[tuple([ind[0][n],ind[1][n],bc[n,-1]]) for n in range(0,len(ind[0]))]
    report['boxcox']['shift']=minR
    report['boxcox']['shited_to']=1

    return R_bc,report

def postpro_ztransform(R,report={}):
    # First make trailing dimension nodal.
    R = np.transpose(R,[2,0,1])
    Z = (R - R.mean(axis=0)) / R.std(axis=0)
    Z = np.transpose(Z,[1,2,0])
    report['ztransform']={}
    report['ztransform']['performed'] = 'yes'
    report['ztransform']['method'] = 'starndard score'
    return Z,report

def postpro_pipeline(R,pipeline,report={}):

    """

    :PARAMETERS:

    :R: pearson correlation value in temporal matrix form (node,node,time)
    :pipeline: list or string (if string, each steps seperated by + sign).

        :options: 'fischer','boxcox','ztransform'

        Each of the above 3 can be specified. If fischer is used, it must be before boxcox. If ztransform is used it must be after boxcox and fischer.

    :report: default is empyt. If non-empty, appended to report.

    :OUTPUT:

    :dataOut: postprocessed data
    :postprocessing_info: dictionary of information about postprocessing (e.g lambda parameters for boxcox)

    """

    if isinstance(pipeline,str):
        if pipeline == "all":
            pipeline = "fisher+boxcox+ztransform"
        pipeline=pipeline.split('+')

    report['postprocess'] = []
    for p in pipeline:
        report['postprocess'].append(p)
        R,report = eval('teneto.derive.postpro_' + p + '(R,report)')
    return R,report
