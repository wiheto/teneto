import numpy as np
from teneto.utils import *

def intercontacttimes(netIn):
    """
    Calculates the intercontacttimes of each edge in a network

    **PARAMETERS**

    :netIn: Temporal network (craphlet or contact).

        :nettype: 'bu', 'bd'

    **OUTPUT**

    :ICT: intercontact times as numpy array

        :format: dictionary

    **NOTES**

    Connections are assumed to be binary

    **SEE ALSO**

    *bursty_coeff*

    **History**

    :Modified: Dec 2016, WHT
    :Created: Nov 2016, WHT

    """


    #Process input
    netIn,netInfo = process_input(netIn,['C','G','TO'])


    if netInfo['nettype'][0]=='d':
        print('WARNING: assuming connections to be binary when computing intercontacttimes')

    #Each time series is padded with a 0 at the start and end. Then t[0:-1]-[t:]. Then discard the noninformative ones (done automatically)
    #Finally return back as np array
    ICT=np.array([[None]*netInfo['netshape'][0]]*netInfo['netshape'][1])

    if netInfo['nettype'][1] == 'u':
        for i in range(0,netInfo['netshape'][0]):
            for j in range(i+1,netInfo['netshape'][0]):
                Aon=np.where(netIn[i,j,:]>0)[0]
                Aon=np.append(0,Aon)
                Aon=np.append(Aon,0)
                Aon_diff=Aon[2:-1]-Aon[1:-2]
                ICT[i,j]=np.array(Aon_diff)
                ICT[j,i]=np.array(Aon_diff)
    elif netInfo['nettype'][1] == 'd':
        for i in range(0,netInfo['netshape'][0]):
            for j in range(0,netInfo['netshape'][0]):
                Aon=np.where(netIn[i,j,:]>0)[0]
                Aon=np.append(0,Aon)
                Aon=np.append(Aon,0)
                Aon_diff=Aon[2:-1]-Aon[1:-2]
                ICT[i,j]=np.array(Aon_diff)

    out={}
    out['intercontacttimes'] = ICT
    out['nettype'] = netInfo['nettype']
    return out
