import numpy as np
from teneto.utils import *

def intercontacttimes(netIn):
    """
    Calculates the temporal degree

    Parameters
    ----------
    netIn: Temporal graph of format (can be bd,bu):
        (i) G: graphlet (3D numpy array).
        (ii) C: contact (dictionary)

    Returns
    ----------
    intercontact times
    format: 1d numpy array

    See Also
    ----------
    burstycoeff

    NOTES
    ----------
    Connections are assumed to be binary

    History
    ----------
    Created - Nov 2016, WHT
    """


    #Get input type (C or G)
    inputType=checkInput(netIn)
    nettype = 'xx'
    #Convert C representation to G
    if inputType == 'C':
        nettype = netIn['nettype']
        netIn = contact2graphlet(netIn)

    #Get network type if not set yet
    if nettype == 'xx':
        nettype = gen_nettype(netIn)

    if nettype[0]=='d':
        print('WARNING: assuming connections to be binary when computing intercontacttimes')

    #Each time series is padded with a 0 at the start and end. Then t[0:-1]-[t:]. Then discard the noninformative ones (done automatically)
    #Finally return back as np array
    ICT=np.array([[None]*netIn.shape[0]]*netIn.shape[1])

    if nettype[1] == 'u':
        for i in range(0,netIn.shape[0]):
            for j in range(i+1,netIn.shape[0]):
                Aon=np.where(netIn[i,j,:]>0)[0]
                Aon=np.append(0,Aon)
                Aon=np.append(Aon,0)
                Aon_diff=Aon[2:-1]-Aon[1:-2]
                ICT[i,j]=np.array(Aon_diff)
                ICT[j,i]=np.array(Aon_diff)
    elif nettype[1] == 'd':
        for i in range(0,netIn.shape[0]):
            for j in range(0,netIn.shape[0]):
                Aon=np.where(netIn[i,j,:]>0)[0]
                Aon=np.append(0,Aon)
                Aon=np.append(Aon,0)
                Aon_diff=Aon[2:-1]-Aon[1:-2]
                ICT[i,j]=np.array(Aon_diff)

    out={}
    out['intercontacttimes'] = ICT
    out['nettype'] = nettype
    return out
